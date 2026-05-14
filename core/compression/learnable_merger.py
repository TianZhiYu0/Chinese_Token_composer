"""
可学习Token合并模块
支持端到端训练，冻结BERT，仅优化合并参数
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class LearnableTokenMerger(nn.Module):
    """
    可学习Token合并器
    1. 通过可学习投影降低维度，计算相似度
    2. 基于可学习阈值决定合并分组
    3. 输出合并后的Token序列（保留顺序）
    """
    def __init__(self, hidden_size=768, proj_dim=128, temperature=0.1):
        super().__init__()
        self.proj = nn.Linear(hidden_size, proj_dim)
        self.temperature = temperature
        # 可学习阈值（sigmoid后映射到[0,1]）
        self.threshold_raw = nn.Parameter(torch.tensor(0.0))

    def forward(self, hidden_states, attention_mask):
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            attention_mask: (batch, seq_len)
        Returns:
            merged_hidden: (batch, new_seq_len, hidden_size)
            merged_mask: (batch, new_seq_len)
            kept_indices: 保留Token的原始索引（用于调试）
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        device = hidden_states.device

        # 1. 投影并归一化
        proj_hidden = self.proj(hidden_states)  # (B, S, D_proj)
        proj_norm = F.normalize(proj_hidden, p=2, dim=-1)

        # 2. 计算相似度矩阵
        sim = torch.matmul(proj_norm, proj_norm.transpose(1, 2))  # (B, S, S)
        # 屏蔽自身和对角线
        eye_mask = torch.eye(seq_len, device=device).bool().unsqueeze(0)
        sim = sim.masked_fill(eye_mask, -1e9)

        # 3. 可学习阈值
        threshold = torch.sigmoid(self.threshold_raw)  # [0,1]

        # 4. 基于阈值和相似度进行软合并（使用连续加权平均，保证梯度可回传）
        # 构建合并权重矩阵：相似度高于阈值的Token相互加权
        sim_weight = torch.softmax(sim / self.temperature, dim=-1)  # (B, S, S)
        # 只对相似度 > threshold 的进行加权，否则权重为0
        mask = (sim > threshold).float()
        sim_weight = sim_weight * mask
        # 归一化（加上自身）
        sim_weight = sim_weight / (sim_weight.sum(dim=-1, keepdim=True) + 1e-8)

        # 5. 加权合并
        merged_hidden = torch.bmm(sim_weight, hidden_states)  # (B, S, D)

        # 6. 去除重复的合并结果（按顺序保留第一个出现的组代表）
        # 此处简化为：如果某Token的合并结果与前面已保留的Token高度相似，则丢弃
        kept_indices = self._deduplicate_merged(merged_hidden, attention_mask, threshold)

        # 构建新的序列
        merged_list = []
        mask_list = []
        for b in range(batch_size):
            cur_hidden = merged_hidden[b, kept_indices[b]]
            cur_mask = attention_mask[b, kept_indices[b]]
            merged_list.append(cur_hidden)
            mask_list.append(cur_mask)

        # 填充到相同长度
        max_len = max(h.shape[0] for h in merged_list)
        padded_hidden = torch.zeros(batch_size, max_len, hidden_size, device=device)
        padded_mask = torch.zeros(batch_size, max_len, device=device)
        for b in range(batch_size):
            cur_len = merged_list[b].shape[0]
            padded_hidden[b, :cur_len] = merged_list[b]
            padded_mask[b, :cur_len] = mask_list[b]

        return padded_hidden, padded_mask, kept_indices

    def _deduplicate_merged(self, merged_hidden, attention_mask, threshold):
        """去重：保留序列中第一个出现的非冗余Token"""
        batch_size, seq_len, _ = merged_hidden.shape
        device = merged_hidden.device
        kept_indices = []
        for b in range(batch_size):
            used = torch.zeros(seq_len, dtype=torch.bool, device=device)
            kept = []
            for i in range(seq_len):
                if not attention_mask[b, i] or used[i]:
                    continue
                kept.append(i)
                used[i] = True
                # 标记后续与i相似的Token为已使用
                for j in range(i + 1, seq_len):
                    if attention_mask[b, j] and not used[j]:
                        sim_ij = F.cosine_similarity(
                            merged_hidden[b, i].unsqueeze(0),
                            merged_hidden[b, j].unsqueeze(0)
                        )
                        if sim_ij > threshold:
                            used[j] = True
            kept_indices.append(torch.tensor(kept, device=device))
        return kept_indices


def create_learnable_merger(model_path: str, device: str = "cpu", **kwargs):
    """加载BERT并构建可学习合并器"""
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    bert = AutoModel.from_pretrained(model_path, local_files_only=True).to(device)
    for param in bert.parameters():
        param.requires_grad = False  # 冻结BERT
    bert.eval()
    merger = LearnableTokenMerger(**kwargs).to(device)
    return tokenizer, bert, merger