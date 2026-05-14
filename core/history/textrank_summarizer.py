import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from typing import List


class TextRankSummarizer:
    """基于 TextRank 的无监督摘要生成器"""

    def __init__(self, encoder_model, num_sentences: int = 3):
        self.encoder = encoder_model
        self.num_sentences = num_sentences

    def _split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'[。！？!?]', text)
        return [s.strip() for s in sentences if s.strip()]

    def _pagerank(self, sim_matrix: np.ndarray, damping: float = 0.85,
                  max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
        n = sim_matrix.shape[0]
        if n == 0:
            return np.array([])
        row_sums = sim_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        transition = sim_matrix / row_sums

        scores = np.ones(n) / n
        for _ in range(max_iter):
            new_scores = (1 - damping) / n + damping * transition.T @ scores
            if np.linalg.norm(new_scores - scores) < tol:
                break
            scores = new_scores
        return scores

    def summarize(self, text: str) -> str:
        sentences = self._split_sentences(text)
        if len(sentences) <= self.num_sentences:
            return text

        embs = self.encoder.encode(sentences)
        sim_matrix = cosine_similarity(embs)
        np.fill_diagonal(sim_matrix, 0)

        scores = self._pagerank(sim_matrix)
        top_indices = np.argsort(scores)[-self.num_sentences:][::-1]
        top_indices = sorted(top_indices)
        return "。".join([sentences[i] for i in top_indices]) + "。"