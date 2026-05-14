import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
import pickle

class VectorStore:
    """
    轻量级向量数据库，用于存储对话摘要
    """
    def __init__(self, encoder_model, dimension=384, persist_path=None):
        self.encoder = encoder_model
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)   # 内积索引（需归一化）
        self.texts = []
        self.persist_path = persist_path
        if persist_path and os.path.exists(persist_path):
            self.load()

    def add(self, text, vector=None):
        """添加文本及其向量"""
        if vector is None:
            vector = self.encoder.encode([text])[0]
        # 归一化
        vector = vector / np.linalg.norm(vector)
        self.index.add(vector.reshape(1, -1))
        self.texts.append(text)

    def search(self, query, top_k=3):
        """检索最相似的 top_k 个文本"""
        query_vec = self.encoder.encode([query])[0]
        query_vec = query_vec / np.linalg.norm(query_vec)
        scores, indices = self.index.search(query_vec.reshape(1, -1), top_k)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx != -1:
                results.append((self.texts[idx], score))
        return results

    def save(self):
        """持久化到文件"""
        if self.persist_path:
            data = {
                'index': faiss.serialize_index(self.index),
                'texts': self.texts
            }
            with open(self.persist_path, 'wb') as f:
                pickle.dump(data, f)

    def load(self):
        """从文件加载"""
        if self.persist_path and os.path.exists(self.persist_path):
            with open(self.persist_path, 'rb') as f:
                data = pickle.load(f)
            self.index = faiss.deserialize_index(data['index'])
            self.texts = data['texts']

    def clear(self):
        """清空所有数据"""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.texts = []