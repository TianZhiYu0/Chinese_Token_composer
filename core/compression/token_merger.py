import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple, Optional


class InferenceTokenMerger(nn.Module):
    """
    推理时Token合并模块：基于余弦相似度动态合并冗余Token。
    该模块插入在BERT编码器之后，对最后一层的隐藏状态进行合并。
    合并后的表示无法直接解码为文本，适用于后续的向量检索或软压缩。
    """
    def __init__(
        self,
        bert_model: nn.Module,
        tokenizer: AutoTokenizer,
        sim_threshold: float = 0.85,
        reduction_ratio: Optional[float] = None,
    ):
        """
        Args:
            bert_model: 已加载的BERT模型（用于获取隐藏状态）
            tokenizer: 对应的分词器（用于识别特殊Token）
            sim_threshold: 余弦相似度阈值，高于此值的Token对将被合并
            reduction_ratio: 目标Token减少比例（如果提供，则动态调整阈值以实现该比例）
        """
        super().__init__()
        self.bert = bert_model
        self.tokenizer = tokenizer
        self.sim_threshold = sim_threshold
        self.reduction_ratio = reduction_ratio

        # 获取特殊Token ID
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id
        self.pad_token_id = tokenizer.pad_token_id

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_dict: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播，返回合并后的Token表示和对应的掩码。

        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)

        Returns:
            merged_hidden: (batch_size, new_seq_len, hidden_size) 合并后的隐藏状态
            merged_mask: (batch_size, new_seq_len) 对应的注意力掩码
            kept_indices: (batch_size, new_seq_len) 保留Token的原始索引（可选）
        """
        # 1. 通过BERT获取最后一层隐藏状态
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # (batch, seq_len, hidden_size)

        # 2. 对每个批次样本进行合并
        batch_size, seq_len, hidden_size = hidden_states.shape
        device = hidden_states.device

        merged_hidden_list = []
        merged_mask_list = []
        kept_indices_list = []

        for b in range(batch_size):
            # 获取有效Token掩码（排除[PAD]）
            valid_mask = attention_mask[b].bool()
            # 获取当前样本的隐藏状态和有效部分
            cur_hidden = hidden_states[b][valid_mask]  # (valid_len, hidden_size)
            cur_input_ids = input_ids[b][valid_mask]   # (valid_len,)
            valid_len = cur_hidden.size(0)

            if valid_len <= 2:  # 只有[CLS]和[SEP]，无需合并
                merged_hidden_list.append(cur_hidden)
                merged_mask_list.append(torch.ones(valid_len, device=device))
                kept_indices_list.append(torch.arange(valid_len, device=device))
                continue

            # 3. 计算有效Token之间的余弦相似度
            normed = F.normalize(cur_hidden, p=2, dim=-1)
            sim_matrix = torch.matmul(normed, normed.transpose(0, 1))  # (valid_len, valid_len)

            # 4. 根据相似度合并Token
            # 标记哪些Token已被合并（仅处理一次）
            used = torch.zeros(valid_len, dtype=torch.bool, device=device)
            # 特殊Token（[CLS]和[SEP]）不参与合并，且始终保留
            special_mask = (cur_input_ids == self.cls_token_id) | (cur_input_ids == self.sep_token_id)
            used[special_mask] = True  # 标记为已处理，避免被合并

            merged_tokens = []
            kept_indices = []

            # 动态调整阈值以接近目标压缩比
            current_threshold = self.sim_threshold
            if self.reduction_ratio is not None:
                # 简单二分搜索，找到一个阈值使保留Token数接近目标
                current_threshold = self._find_threshold_for_ratio(
                    sim_matrix, special_mask, valid_len, self.reduction_ratio
                )

            for i in range(valid_len):
                if used[i]:
                    # 如果是特殊Token，直接保留
                    if special_mask[i]:
                        merged_tokens.append(cur_hidden[i])
                        kept_indices.append(i)
                    continue

                # 找到所有与i相似且未被使用的Token
                similar = (sim_matrix[i] > current_threshold) & ~used & ~special_mask
                if similar.sum() > 1:
                    # 合并相似Token：取它们的平均值
                    merged_token = cur_hidden[similar].mean(dim=0)
                    merged_tokens.append(merged_token)
                    # 记录第一个Token的索引作为代表
                    rep_idx = torch.where(similar)[0][0].item()
                    kept_indices.append(rep_idx)
                    used[similar] = True
                else:
                    merged_tokens.append(cur_hidden[i])
                    kept_indices.append(i)
                    used[i] = True

            merged_hidden = torch.stack(merged_tokens)  # (new_len, hidden_size)
            merged_mask = torch.ones(merged_hidden.size(0), device=device)

            merged_hidden_list.append(merged_hidden)
            merged_mask_list.append(merged_mask)
            kept_indices_list.append(torch.tensor(kept_indices, device=device))

        # 5. 将结果填充到相同长度
        max_len = max(h.shape[0] for h in merged_hidden_list)
        padded_hidden = torch.zeros(batch_size, max_len, hidden_size, device=device)
        padded_mask = torch.zeros(batch_size, max_len, device=device)
        padded_indices = torch.full((batch_size, max_len), -1, device=device, dtype=torch.long)

        for b in range(batch_size):
            cur_len = merged_hidden_list[b].shape[0]
            padded_hidden[b, :cur_len] = merged_hidden_list[b]
            padded_mask[b, :cur_len] = merged_mask_list[b]
            padded_indices[b, :cur_len] = kept_indices_list[b]

        if return_dict:
            return {
                "merged_hidden": padded_hidden,
                "merged_mask": padded_mask,
                "kept_indices": padded_indices,
            }
        return padded_hidden, padded_mask, padded_indices

    def _find_threshold_for_ratio(
        self, sim_matrix: torch.Tensor, special_mask: torch.Tensor, valid_len: int, target_ratio: float
    ) -> float:
        """
        简单的二分搜索，找到合适的阈值，使保留Token数接近 target_ratio * valid_len。
        """
        lo, hi = 0.0, 1.0
        best_thresh = self.sim_threshold
        best_diff = float('inf')
        target_keep = int(valid_len * (1 - target_ratio))  # 目标保留数

        for _ in range(10):  # 10次二分足够
            mid = (lo + hi) / 2
            keep_count = self._simulate_keep_count(sim_matrix, special_mask, mid)
            diff = abs(keep_count - target_keep)
            if diff < best_diff:
                best_diff = diff
                best_thresh = mid
            if keep_count > target_keep:
                hi = mid
            else:
                lo = mid
        return best_thresh

    def _simulate_keep_count(self, sim_matrix: torch.Tensor, special_mask: torch.Tensor, threshold: float) -> int:
        """模拟在给定阈值下，最终保留的Token数量"""
        used = special_mask.clone()
        keep_count = special_mask.sum().item()
        for i in range(sim_matrix.size(0)):
            if used[i]:
                continue
            similar = (sim_matrix[i] > threshold) & ~used & ~special_mask
            if similar.sum() > 1:
                keep_count += 1
                used[similar] = True
            else:
                keep_count += 1
                used[i] = True
        return keep_count


# ---------- 便捷接口：直接加载模型并创建合并器 ----------
def create_token_merger(model_path: str, device: str = "cpu", **kwargs) -> InferenceTokenMerger:
    """
    加载BERT模型并创建推理时Token合并器。

    Args:
        model_path: 本地BERT模型路径
        device: 运行设备
        **kwargs: 传递给InferenceTokenMerger的参数（如sim_threshold, reduction_ratio）
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    bert = AutoModel.from_pretrained(model_path, local_files_only=True).to(device)
    bert.eval()
    return InferenceTokenMerger(bert, tokenizer, **kwargs)