import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from core.utils.utils import count_tokens


class HardCompressor:
    def __init__(self, model_path=None, model_type="mooscomp", device=None, batch_size=32):
        """
        初始化硬压缩器
        
        参数:
            model_path: 模型路径（如果为None，则根据model_type自动选择）
            model_type: "mooscomp" 或 "llmlingua2"
            device: 运行设备
            batch_size: 批处理大小
        """
        import os
        self.model_type = model_type

        # 自动确定模型路径
        if model_path is None:
            if model_type == "mooscomp":
                model_path = "model/compression_bert_mooscomp_news"
            elif model_type == "llmlingua2":
                model_path = "model/llmlingua-2-bert-base-multilingual-cased-meetingbank"
            else:
                raise ValueError(f"未知的压缩模型类型: {model_type}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型路径不存在: {model_path}")

        # 保存 model_path 属性（供 preprocess_learnable_merge 等方法使用）
        self.model_path = model_path

        print(f"正在加载 BERT 压缩模型 ({model_type}): {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True
        )
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_path,
            local_files_only=True
        )

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer.model_max_length = 512
        self.batch_size = batch_size
        print(f"BERT 压缩模型加载成功，设备: {self.device}")

    def compress(self, sentence, keep_label_id=1, min_span_length=2, compression_ratio=None):
        """
        单句压缩：基于概率排序保留重要token，返回 (压缩文本, 原始token数, 压缩后token数)
        """
        if not sentence:
            return "", 0, 0
        if compression_ratio is None or compression_ratio >= 1.0:
            orig_tokens = count_tokens(self.tokenizer, sentence)
            return sentence, orig_tokens, orig_tokens

        inputs = self.tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0, :, keep_label_id]
        input_ids = inputs["input_ids"][0]
        seq_len = input_ids.size(0)

        # 有效token（排除特殊标记）
        special_ids = {self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id}
        valid_mask = torch.tensor([tid.item() not in special_ids for tid in input_ids], device=self.device)
        valid_indices = torch.where(valid_mask)[0]

        if len(valid_indices) == 0:
            return "", 0, 0

        target_keep = max(1, int(len(valid_indices) * compression_ratio))

        # 按概率降序选择保留token
        valid_probs = probs[valid_indices]
        _, sorted_idx = torch.sort(valid_probs, descending=True)
        keep_indices_in_valid = sorted_idx[:target_keep]
        keep_global_indices = valid_indices[keep_indices_in_valid]

        # 构造保留集合，始终包含 [CLS] 和 [SEP] 以保证解码正确
        keep_set = set(keep_global_indices.tolist())
        keep_set.add(0)
        keep_set.add(seq_len - 1)

        kept_ids = [input_ids[i].item() for i in range(seq_len) if i in keep_set]

        # 解码并清理空格
        compressed = self.tokenizer.decode(
            kept_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        ).replace(" ", "")

        original_tokens = count_tokens(self.tokenizer, sentence)
        compressed_tokens = count_tokens(self.tokenizer, compressed)
        return compressed, original_tokens, compressed_tokens

    def compress_batch(self, sentences, keep_label_id=1, min_span_length=2, compression_ratio=None):
        """
        批量压缩，返回压缩后的字符串列表
        """
        if not sentences:
            return []
        if compression_ratio is None or compression_ratio >= 1.0:
            return sentences

        inputs = self.tokenizer(
            sentences,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[:, :, keep_label_id]

        compressed_list = []
        for i in range(len(sentences)):
            input_ids = inputs["input_ids"][i]
            seq_len = input_ids.size(0)
            prob_i = probs[i]

            valid_mask = (input_ids != self.tokenizer.cls_token_id) & \
                         (input_ids != self.tokenizer.sep_token_id) & \
                         (input_ids != self.tokenizer.pad_token_id)
            valid_indices = torch.where(valid_mask)[0]

            if len(valid_indices) == 0:
                compressed_list.append("")
                continue

            target_keep = max(1, int(len(valid_indices) * compression_ratio))

            valid_probs = prob_i[valid_indices]
            _, sorted_idx = torch.sort(valid_probs, descending=True)
            keep_indices_in_valid = sorted_idx[:target_keep]
            keep_global_indices = valid_indices[keep_indices_in_valid]

            keep_set = set(keep_global_indices.tolist())
            keep_set.add(0)
            keep_set.add(seq_len - 1)

            kept_ids = [input_ids[j].item() for j in range(seq_len) if j in keep_set]
            compressed = self.tokenizer.decode(
                kept_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            ).replace(" ", "")
            compressed_list.append(compressed)

        return compressed_list


    # 以下方法保留原有接口，但不再用于主要压缩流程
    def _extract_continuous_spans(self, input_ids, preds, keep_label_id, min_span_length):
        kept_ids = []
        current_span = []
        for tid, label in zip(input_ids, preds):
            if tid in [0, 101, 102]:
                if len(current_span) >= min_span_length:
                    kept_ids.extend(current_span)
                current_span = []
                continue
            if label == keep_label_id:
                current_span.append(tid)
            else:
                if len(current_span) >= min_span_length:
                    kept_ids.extend(current_span)
                current_span = []
        if len(current_span) >= min_span_length:
            kept_ids.extend(current_span)
        return kept_ids

    def _extract_with_ratio_control(self, input_ids, preds, keep_label_id, target_ratio, min_span_length):
        input_tensor = torch.tensor([input_ids]).to(self.device)
        with torch.no_grad():
            logits = self.model(input_ids=input_tensor).logits
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        candidates = []
        for idx, (tid, label) in enumerate(zip(input_ids, preds)):
            if tid in [0, 101, 102]:
                continue
            if label == keep_label_id:
                confidence = probs[idx][keep_label_id]
                candidates.append((tid, confidence, idx))
        if not candidates:
            return []
        total_content = sum(1 for tid in input_ids if tid not in [0, 101, 102])
        target_count = max(1, int(total_content * target_ratio))
        candidates.sort(key=lambda x: x[1], reverse=True)
        selected = candidates[:target_count]
        selected.sort(key=lambda x: x[2])
        if min_span_length > 1:
            return self._filter_by_span([tid for tid, _, _ in selected], min_span_length)
        return [tid for tid, _, _ in selected]

    def compress_to_target_tokens(self, text: str, target_tokens: int, chunk_size: int = 256) -> str:
        """
        将超长文本压缩到目标 token 数。
        采用分块推理+全局Token选择的策略，避免分块解码拼接导致的膨胀。
        
        核心改进：
        1. 分块推理获取概率，但保留原始Token ID序列
        2. 全局按概率选择Top-K Token
        3. 按原始位置排序后，一次解码，避免中间文本拼接
        
        参数:
            text: 输入文本
            target_tokens: 目标 token 数量
            chunk_size: 每个块的最大 token 数（默认256）
            
        返回:
            压缩后的文本
        """
        if not text:
            return ""
        
        # 1. 先对全文编码，获取原始Token ID序列
        full_encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=False,  # 不截断，保留全部
            add_special_tokens=False  # 不添加特殊token
        ).to(self.device)
        
        full_input_ids = full_encoding["input_ids"][0]
        original_token_count = len(full_input_ids)
        
        # 如果原文本Token数已经小于目标值，直接返回原文
        if original_token_count <= target_tokens:
            return text
        
        # 2. 分块推理获取概率（避免超出模型max_length限制）
        all_probs = []
        total_len = len(full_input_ids)
        
        for start_idx in range(0, total_len, chunk_size):
            end_idx = min(start_idx + chunk_size, total_len)
            chunk_ids = full_input_ids[start_idx:end_idx]
            
            # 推理获取概率
            with torch.no_grad():
                outputs = self.model(input_ids=chunk_ids.unsqueeze(0))
            
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0, :, 1]  # 保留类概率（label_id=1）
            all_probs.append(probs)
        
        # 合并所有块的概率
        all_probs = torch.cat(all_probs, dim=0)
        
        # 3. 全局按概率排序，选择Top-K Token
        # 创建(概率, 位置)的元组列表
        token_scores = []
        for i in range(original_token_count):
            tid = full_input_ids[i].item()
            prob = all_probs[i].item()
            token_scores.append((prob, i, tid))  # (概率, 位置, Token ID)
        
        # 按概率降序排序
        token_scores.sort(key=lambda x: x[0], reverse=True)
        
        # 选择Top-K
        selected = token_scores[:target_tokens]
        
        # 4. 按原始位置排序（保持文本顺序）
        selected.sort(key=lambda x: x[1])  # 按位置排序
        
        # 提取Token ID
        selected_ids = [tid for _, _, tid in selected]
        
        # 5. 一次解码（避免分块解码拼接问题）
        compressed_text = self.tokenizer.decode(
            selected_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        # 6. 验证压缩后Token数
        compressed_token_count = len(self.tokenizer.encode(
            compressed_text,
            add_special_tokens=False
        ))
        
        # 如果因解码问题导致Token数超出目标，循环减少直至满足
        while compressed_token_count > target_tokens and len(selected_ids) > 1:
            # 移除概率最低的Token
            selected.pop()  # 移除最后一个（概率最低）
            selected_ids = [tid for _, _, tid in selected]
            compressed_text = self.tokenizer.decode(
                selected_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            compressed_token_count = len(self.tokenizer.encode(
                compressed_text,
                add_special_tokens=False
            ))
        
        return compressed_text

    def _filter_by_span(self, token_ids, min_span_length):
        if not token_ids:
            return []
        kept = []
        current_span = []
        for tid in token_ids:
            current_span.append(tid)
            if len(current_span) >= min_span_length:
                kept.extend(current_span)
                current_span = []
        return kept

    def compress_with_confidence(self, sentence, threshold=0.7, keep_label_id=1):
        inputs = self.tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        input_ids = inputs["input_ids"].squeeze(0).cpu().numpy()
        kept_ids = [
            tid for tid, prob in zip(input_ids, probs)
            if tid not in [0, 101, 102] and prob[keep_label_id] > threshold
        ]
        compressed = self.tokenizer.decode(kept_ids, skip_special_tokens=True)
        return compressed