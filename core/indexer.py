import faiss
import numpy as np

class Indexer:
    def __init__(self, dimension):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)   # 内积索引（需要归一化）
        self.vectors = None
        self.fragments = None

    def build(self, vectors, fragments):
        """
        构建索引
        vectors: numpy array, shape (n, dim)
        fragments: list of strings
        """
        self.vectors = vectors.copy()
        self.fragments = fragments.copy()
        faiss.normalize_L2(self.vectors)
        self.index.reset()
        self.index.add(self.vectors)
        return self

    def search(self, query_vector, top_k):
        """
        检索最相似的 top_k 个片段
        返回 (scores, indices)
        """
        if self.index.ntotal == 0:
            return [], []
        query_vec = query_vector.copy()
        faiss.normalize_L2(query_vec)
        scores, indices = self.index.search(query_vec, top_k)
        return scores[0], indices[0]

    def get_fragments_by_indices(self, indices):
        return [self.fragments[i] for i in indices]