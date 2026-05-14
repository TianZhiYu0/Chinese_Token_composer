#!/usr/bin/env python3
"""
混合压缩器 (优化版)：
  直接加载 MOOSComp (token级二分类) 与 WordPriority (词级回归) 模型，
  在 token 级加权融合，再聚合到词级进行压缩，输出连贯的词序列。
"""
import os
import torch
import jieba
import numpy as np
from typing import List
from transformers import BertTokenizerFast, BertForTokenClassification

from core.compression.word_priority_model import WordPriorityModel

class HybridCompressor:
    def __init__(self,
                 bert_model_path: str,
                 word_priority_model_path: str,
                 mooscomp_model_path: str,
                 device: str = "cpu",
                 dynamic_alpha: bool = True,
                 fixed_alpha: float = 0.1):#0.1压缩比下：语义完整性最佳0.1，问答完整性最佳0.9
        """
        Args:
            bert_model_path: BERT 基座路径（两个模型共享，如 .../compression_bert_mooscomp_news）
            word_priority_model_path: 词级优先级模型权重 .pt 文件路径
            mooscomp_model_path: MOOSComp 模型目录（与 bert_model_path 相同或独立）
            device: 设备
            dynamic_alpha: 是否根据压缩比自动调整 MOOSComp 权重
            fixed_alpha: 若 dynamic_alpha=False，则使用此固定权重
        """
        self.device = device
        self.dynamic_alpha = dynamic_alpha
        self.fixed_alpha = fixed_alpha

        # 加载 Tokenizer
        self.tokenizer = BertTokenizerFast.from_pretrained(bert_model_path, local_files_only=True)

        # 1. 加载 MOOSComp 模型 (token 级二分类，保留/丢弃)
        self.mooscomp_model = BertForTokenClassification.from_pretrained(
            mooscomp_model_path,
            local_files_only=True
        )
        self.mooscomp_model.to(device)
        self.mooscomp_model.eval()

        # 2. 加载 WordPriority 模型
        self.word_model = WordPriorityModel(bert_model_path=bert_model_path)
        checkpoint = torch.load(word_priority_model_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        self.word_model.load_state_dict(state_dict, strict=False)
        self.word_model.to(device)
        self.word_model.eval()

    @torch.no_grad()
    def compress(self, text: str, compression_ratio: float) -> str:
        """混合压缩单条文本，返回压缩后的词序列拼接字符串"""
        # 1. 分词
        words = list(jieba.cut(text))
        n_words = len(words)
        if n_words == 0:
            return ""

        # 2. Tokenize 并获取 offset mapping
        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)

        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        offset_mapping = encoding['offset_mapping'][0].cpu().tolist()

        # 3. 构建 word_to_tokens 映射
        word_spans = []
        pos = 0
        for w in words:
            start = pos
            end = pos + len(w)
            word_spans.append((start, end))
            pos = end

        word_to_tokens = [[] for _ in range(n_words)]
        for token_idx, (char_start, char_end) in enumerate(offset_mapping):
            if char_start == 0 and char_end == 0:  # [CLS], [SEP], [PAD]
                continue
            # 查找该 token 属于哪个词
            for w_idx, (w_start, w_end) in enumerate(word_spans):
                if char_start >= w_start and char_start < w_end:
                    word_to_tokens[w_idx].append(token_idx)
                    break

        # 4. 获取 MOOSComp token 级分数 (保留概率)
        moos_outputs = self.mooscomp_model(input_ids, attention_mask=attention_mask)
        moos_logits = moos_outputs.logits  # (1, seq_len, num_labels)
        moos_probs = torch.softmax(moos_logits, dim=-1)[0, :, 1]  # 标签1为保留
        moos_scores = moos_probs.cpu().numpy()

        # 5. 获取 WordPriority 词级分数
        # 需要准备 word_boundaries：将 token 索引映射到词内位置，用于模型内部池化
        # WordPriorityModel.forward 接受 word_boundaries: List[(start, end)]
        # 我们直接构建边界列表（基于 input_ids）
        word_boundaries = []
        for token_indices in word_to_tokens:
            if token_indices:
                word_boundaries.append((token_indices[0], token_indices[-1] + 1))
            else:
                word_boundaries.append((0, 0))  # 占位
        # 转换 word_boundaries 为模型需要的格式，注意 batch 维度
        word_outputs, _ = self.word_model(input_ids, attention_mask, [word_boundaries])
        # word_outputs: (1, num_words) 或 (1, max_words)，我们需要实际词数
        word_scores = word_outputs[0, :n_words].cpu().numpy()

        # 6. 将词级分数映射到 token 级 (取词分数)
        word_token_scores = np.zeros(len(moos_scores), dtype=np.float32)
        for w_idx, score in enumerate(word_scores):
            for tok_idx in word_to_tokens[w_idx]:
                word_token_scores[tok_idx] = score

        # 7. 加权融合
        alpha = self._get_alpha(compression_ratio)
        combined_token_scores = alpha * moos_scores + (1 - alpha) * word_token_scores

        # 8. 聚合回词级分数 (均值)
        word_combined_scores = np.zeros(n_words, dtype=np.float32)
        for w_idx, token_indices in enumerate(word_to_tokens):
            if token_indices:
                word_combined_scores[w_idx] = np.mean(combined_token_scores[token_indices])
            else:
                word_combined_scores[w_idx] = 0.0

        # 9. 选择 Top-K 词
        k = max(1, int(n_words * compression_ratio))
        top_indices = np.argsort(word_combined_scores)[-k:]

        # 10. 按原文顺序输出词
        kept_words = [words[i] for i in range(n_words) if i in top_indices]
        return ''.join(kept_words)

    def compress_batch(self, texts: List[str], compression_ratio: float) -> List[str]:
        """批量压缩"""
        return [self.compress(t, compression_ratio) for t in texts]

    def _get_alpha(self, compression_ratio: float) -> float:
        """动态计算 MOOSComp 权重"""
        if not self.dynamic_alpha:
            return self.fixed_alpha
        if compression_ratio <= 0.3:
            return 0.95
        elif compression_ratio <= 0.5:
            return 0.85
        elif compression_ratio <= 0.7:
            return 0.80
        else:
            return 0.75