"""
增强型层次化压缩模块
整合：问题感知筛选、段落摘要、动态压缩率、文档重排序
"""
import numpy as np
from typing import List, Optional

from core.compression.compressor import HardCompressor
from core.engine.summarizer import DialogSummarizer
from core.utils.utils import split_sentences


class EnhancedHierarchicalCompressor:
    def __init__(
        self,
        model_path: str,
        encoder,
        summarizer: DialogSummarizer,
        device: str = "cpu",
        relevance_threshold: float = 0.3,
        base_compression_ratio: float = 0.7,
        high_ratio: float = 0.85,
        low_ratio: float = 0.4,
    ):
        """
        Args:
            model_path: BERT硬压缩模型路径
            encoder: Sentence-Transformer编码器（用于计算相似度）
            summarizer: 对话摘要器（用于生成段落摘要）
            device: 运行设备
            relevance_threshold: 段落相关性阈值，低于此值的段落被丢弃
            base_compression_ratio: 基础压缩率（无问题感知时使用）
            high_ratio: 高相关段落的压缩率（保留更多）
            low_ratio: 低相关段落的压缩率（保留更少）
        """
        self.hard_compressor = HardCompressor(model_path, device=device)
        self.encoder = encoder
        self.summarizer = summarizer
        self.relevance_threshold = relevance_threshold
        self.base_ratio = base_compression_ratio
        self.high_ratio = high_ratio
        self.low_ratio = low_ratio

    def compress_documents(
        self,
        documents: List[str],
        question: Optional[str] = None,
    ) -> List[str]:
        """
        对文档列表进行增强压缩

        Args:
            documents: 原始文档列表（每个元素是一篇文档或一个长段落）
            question: 用户问题（若为None，则使用无问题感知的均匀压缩）

        Returns:
            压缩后的文本片段列表（已重排序）
        """
        if not documents:
            return []

        # 1. 将文档拆分为段落（若已是段落则跳过）
        all_paragraphs = []
        for doc in documents:
            paras = self._split_into_paragraphs(doc)
            all_paragraphs.extend(paras)

        # 2. 问题感知粗粒度筛选 + 动态压缩率分配
        if question:
            para_scores = self._compute_relevance_scores(all_paragraphs, question)
            kept_paras = []
            kept_scores = []
            for para, score in zip(all_paragraphs, para_scores):
                if score >= self.relevance_threshold:
                    kept_paras.append(para)
                    kept_scores.append(score)
            # 若全部被过滤，则保留分数最高的3个
            if not kept_paras:
                top_indices = np.argsort(para_scores)[-3:][::-1]
                kept_paras = [all_paragraphs[i] for i in top_indices]
                kept_scores = [para_scores[i] for i in top_indices]
        else:
            # 无问题感知：保留所有段落，分数设为 None（触发均匀压缩）
            kept_paras = all_paragraphs
            kept_scores = [None] * len(kept_paras)

        # 3. 为每个保留段落生成摘要
        summarized_paras = []
        for para in kept_paras:
            summary = self._generate_summary(para)
            summarized_paras.append((summary, para))

        # 4. 细粒度压缩（摘要 + 原文，动态压缩率）
        compressed_paras = []
        for (summary, para), score in zip(summarized_paras, kept_scores):
            ratio = self._get_compression_ratio(score)

            # 修复：分别压缩摘要和原文，而非拼接后压缩
            compressed_summary = self.hard_compressor.compress(
                summary, compression_ratio=min(1.0, ratio * 1.2)  # 摘要可稍高保留
            )[0]
            compressed_para = self.hard_compressor.compress(
                para, compression_ratio=ratio
            )[0]

            # 清理可能的摘要标记残留（若摘要本身包含特殊字符）
            compressed_summary = compressed_summary.replace("【", "").replace("】", "")
            # 拼接：摘要在前，原文在后
            combined = f"{compressed_summary} {compressed_para}".strip()
            compressed_paras.append(combined)

        # 5. 文档重排序（沙漏型：高分首尾，低分中间）
        if question:
            reordered = self._reorder_by_importance(compressed_paras, kept_scores)
        else:
            reordered = compressed_paras
        return reordered

    def _split_into_paragraphs(self, text: str, max_para_len: int = 512) -> List[str]:
        """将长文本切分为段落（按换行或句子累积）"""
        raw_paras = [p.strip() for p in text.split('\n') if p.strip()]
        if not raw_paras:
            raw_paras = [text]

        final_paras = []
        for para in raw_paras:
            if len(para) <= max_para_len:
                final_paras.append(para)
            else:
                sentences = split_sentences(para)
                current = ""
                for sent in sentences:
                    if len(current) + len(sent) <= max_para_len:
                        current += sent
                    else:
                        if current:
                            final_paras.append(current)
                        current = sent
                if current:
                    final_paras.append(current)
        return final_paras

    def _compute_relevance_scores(self, paragraphs: List[str], question: str) -> List[float]:
        """计算段落与问题的相似度分数"""
        if not paragraphs:
            return []
        q_vec = self.encoder.encode([question])[0]
        q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-8)
        p_vecs = self.encoder.encode(paragraphs)
        p_vecs = p_vecs / (np.linalg.norm(p_vecs, axis=1, keepdims=True) + 1e-8)
        scores = np.dot(p_vecs, q_vec)
        return scores.tolist()

    def _generate_summary(self, text: str, max_summary_len: int = 64) -> str:
        """生成段落摘要"""
        try:
            summary = self.summarizer.summarize(text, max_length=max_summary_len)
            return summary if summary else text[:max_summary_len]
        except Exception:
            return text[:max_summary_len] + "..."

    def _get_compression_ratio(self, relevance_score: Optional[float]) -> float:
        """
        根据相关性分数动态分配压缩率。
        若分数为 None，表示无问题感知，直接返回基准压缩比。
        """
        if relevance_score is None:
            return self.base_ratio
        clamped_score = max(0.0, min(1.0, relevance_score))
        ratio = self.low_ratio + (self.high_ratio - self.low_ratio) * clamped_score
        return ratio

    def _reorder_by_importance(self, items: List[str], scores: List[float]) -> List[str]:
        """沙漏型重排序：最高分在首尾，次高分依次向内"""
        if len(items) <= 2:
            return items
        paired = sorted(zip(items, scores), key=lambda x: x[1], reverse=True)
        sorted_items = [p[0] for p in paired]
        reordered = []
        left, right = 0, len(sorted_items) - 1
        while left <= right:
            if left == right:
                reordered.append(sorted_items[left])
            else:
                reordered.append(sorted_items[left])
                reordered.append(sorted_items[right])
            left += 1
            right -= 1
        return reordered