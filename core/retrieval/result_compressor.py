import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any

class ResultCompressor:
    """检索结果压缩器：语义去重 + 长度剪枝，无 LLM 依赖"""
    
    def __init__(self, encoder_model):
        """
        encoder_model: SentenceTransformer 编码器（复用 merger.encoder）
        """
        self.encoder = encoder_model
    
    def deduplicate(self, chunks: List[Dict], threshold: float = 0.85) -> List[Dict]:
        """基于语义相似度去除冗余 chunks"""
        if len(chunks) <= 1:
            return chunks
        
        texts = [c["content"] for c in chunks]
        embs = self.encoder.encode(texts)
        
        keep_indices = [0]
        for i in range(1, len(embs)):
            sims = cosine_similarity([embs[i]], embs[keep_indices])[0]
            if max(sims) < threshold:
                keep_indices.append(i)
        
        return [chunks[i] for i in keep_indices]
    
    def prune_by_length(self, chunks: List[Dict], max_total_chars: int = 6000) -> List[Dict]:
        """按总字符数剪枝，优先保留分数高或更长的 chunk"""
        if not chunks:
            return chunks
        
        # 先按分数（若有）降序，再按长度降序
        def sort_key(c):
            score = c.get("score", 0)
            length = len(c.get("content", ""))
            return (score, length)
        
        sorted_chunks = sorted(chunks, key=sort_key, reverse=True)
        result = []
        total_chars = 0
        for chunk in sorted_chunks:
            chunk_len = len(chunk["content"])
            if total_chars + chunk_len <= max_total_chars:
                result.append(chunk)
                total_chars += chunk_len
            else:
                remaining = max_total_chars - total_chars
                if remaining > 200:  # 剩余空间足够放一点内容
                    truncated = chunk["content"][:remaining]
                    chunk["content"] = truncated
                    result.append(chunk)
                break
        return result
    
    def rerank_by_relevance(self, query: str, chunks: List[Dict], 
                           top_k: int = 5) -> List[Dict]:
        """
        基于问题-文档相关性重排序
        
        策略：
        1. 计算问题与每个chunk的语义相似度
        2. 结合原始分数进行加权重排序
        3. 返回top_k个最相关的chunks
        
        Args:
            query: 用户问题
            chunks: 检索到的文档片段列表
            top_k: 返回数量
        
        Returns:
            重排序后的chunks列表
        """
        if not chunks:
            return chunks
        
        # 编码问题和chunks
        query_emb = self.encoder.encode([query])[0]
        chunk_texts = [c["content"] for c in chunks]
        chunk_embs = self.encoder.encode(chunk_texts)
        
        # 计算相似度
        similarities = cosine_similarity([query_emb], chunk_embs)[0]
        
        # 加权分数：原始分数 * 0.4 + 重排序分数 * 0.6
        for i, chunk in enumerate(chunks):
            original_score = chunk.get("score", 0)
            rerank_score = float(similarities[i])
            chunk["rerank_score"] = rerank_score
            chunk["combined_score"] = original_score * 0.4 + rerank_score * 0.6
        
        # 按综合分数排序
        chunks.sort(key=lambda x: x["combined_score"], reverse=True)
        
        # 返回top_k
        return chunks[:top_k]