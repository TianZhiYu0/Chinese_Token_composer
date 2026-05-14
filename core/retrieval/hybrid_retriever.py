"""
混合检索器
==========
整合 BM25 关键词检索与稠密向量检索，支持上下文扩展和文档来源标记

优化策略：
1. BM25 + 稠密向量加权融合（默认偏向 BM25，alpha=0.7）
2. 扩大候选池到100，提高召回率
3. 上下文扩展：扩展核心片段的前后相关内容
4. 文档来源标记：为每个片段添加 [文档N] 前缀
5. 动态扩展机制：当上下文token不足窗口40%时自动扩大检索范围
"""
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import jieba


def tokenize_for_bm25(text: str) -> List[str]:
    """为 BM25 分词（中文优化）"""
    return [w for w in jieba.lcut(text) if len(w.strip()) > 1]


class HybridRetriever:
    """混合检索器：BM25 + 稠密向量 + 上下文扩展"""
    
    def __init__(self, encoder, tokenizer, alpha: float = 0.7):
        """
        Args:
            encoder: SentenceTransformer 编码器
            tokenizer: Tokenizer 用于token计数
            alpha: BM25权重（0-1，越大越偏向BM25）
        """
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.alpha = alpha
        
        # 检索配置
        self.candidate_pool_size = 100  # 候选池大小
        self.min_context_ratio = 0.4    # 最小上下文比例（触发动态扩展）
        self.max_expansion_attempts = 5 # 最大扩展尝试次数
        
    def _compute_bm25_scores(self, fragments: List[str], query: str) -> np.ndarray:
        """计算 BM25 分数"""
        tokenized_corpus = [tokenize_for_bm25(doc) for doc in fragments]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = tokenize_for_bm25(query)
        scores = bm25.get_scores(tokenized_query)
        return np.array(scores)
    
    def _compute_dense_scores(self, vectors: np.ndarray, query: str) -> np.ndarray:
        """计算稠密向量分数"""
        if len(vectors) == 0:
            return np.array([])
        q_emb = self.encoder.encode([query], convert_to_numpy=True)
        return cosine_similarity(q_emb, vectors)[0]
    
    def _fuse_scores(
        self,
        bm25_scores: np.ndarray,
        dense_scores: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """融合 BM25 和稠密向量分数"""
        if len(bm25_scores) == 0:
            return dense_scores
        if len(dense_scores) == 0:
            return bm25_scores

        # 归一化
        bm25_norm = bm25_scores / (bm25_scores.max() + 1e-8)
        dense_norm = dense_scores / (dense_scores.max() + 1e-8)

        return alpha * bm25_norm + (1 - alpha) * dense_norm
    
    def _expand_context_around_indices(
        self,
        core_indices: List[int],
        doc_ids: List[int],
        doc_orders: List[int],
        fragments: List[str],
        context_window: int,
        max_tokens: int
    ) -> Tuple[List[str], int, set]:
        """扩展核心片段的上下文（添加文档来源标记）"""
        expanded_indices = set()

        # 扩展前后上下文
        for idx in core_indices:
            expanded_indices.add(idx)
            doc_id = doc_ids[idx]
            order = doc_orders[idx]

            # 向前扩展
            for d in range(1, context_window + 1):
                prev_order = order - d
                if prev_order >= 0:
                    for i, (did, dord) in enumerate(zip(doc_ids, doc_orders)):
                        if did == doc_id and dord == prev_order:
                            expanded_indices.add(i)
                            break

            # 向后扩展
            for d in range(1, context_window + 1):
                max_order = max(o for did, o in zip(doc_ids, doc_orders) if did == doc_id)
                next_order = order + d
                if next_order <= max_order:
                    for i, (did, dord) in enumerate(zip(doc_ids, doc_orders)):
                        if did == doc_id and dord == next_order:
                            expanded_indices.add(i)
                            break

        # 按文档和顺序排序
        indexed = [(doc_ids[i], doc_orders[i], i, fragments[i]) for i in expanded_indices]
        indexed.sort(key=lambda x: (x[0], x[1]))

        # 拼接上下文（带文档来源标记）
        context_parts = []
        cur_tokens = 0
        used_indices = set()

        for _, _, i, frag in indexed:
            doc_label = f"[文档{doc_ids[i]}]"
            marked_frag = f"{doc_label} {frag}"
            t = self._count_tokens(marked_frag)
            
            if cur_tokens + t > max_tokens:
                remaining = max_tokens - cur_tokens
                if remaining > 50:
                    truncated_ids = self.tokenizer.encode(marked_frag, truncation=True, max_length=remaining)
                    truncated = self.tokenizer.decode(truncated_ids, skip_special_tokens=True)
                    context_parts.append(truncated + "...")
                    used_indices.add(i)
                break
            context_parts.append(marked_frag)
            used_indices.add(i)
            cur_tokens += t

        return context_parts, cur_tokens, used_indices
    
    def _count_tokens(self, text: str) -> int:
        """计算token数"""
        return len(self.tokenizer.encode(text, add_special_tokens=False))
    
    def _single_retrieve(
        self,
        fragments: List[str],
        vectors: np.ndarray,
        doc_ids: List[int],
        doc_orders: List[int],
        question: str,
        top_k: int,
        context_window: int,
        max_tokens: int,
        alpha: float
    ) -> Tuple[str, dict]:
        """单次检索（优化版）"""
        # 计算分数
        bm25_scores = self._compute_bm25_scores(fragments, question)
        dense_scores = self._compute_dense_scores(vectors, question)
        combined = self._fuse_scores(bm25_scores, dense_scores, alpha)

        # 扩大候选池
        candidate_pool_size = min(self.candidate_pool_size, len(fragments))
        candidate_indices = np.argsort(combined)[::-1][:candidate_pool_size].tolist()
        core_indices = candidate_indices[:top_k]

        # 上下文扩展
        context_parts, total_tokens, used_indices = self._expand_context_around_indices(
            core_indices, doc_ids, doc_orders, fragments, context_window, max_tokens
        )

        context = "\n\n".join(context_parts)
        audit = {
            "core_indices": core_indices,
            "candidate_pool_size": len(candidate_indices),
            "expanded_count": len(used_indices),
            "total_tokens": total_tokens,
            "context_window": context_window,
            "top_k": top_k,
        }
        return context, audit
    
    def retrieve(
        self,
        fragments: List[str],
        vectors: np.ndarray,
        doc_ids: List[int],
        doc_orders: List[int],
        question: str,
        top_k: int = 5,
        context_window: int = 2,
        max_tokens: int = 4096,
        alpha: Optional[float] = None
    ) -> Tuple[str, dict]:
        """
        混合检索 + 上下文扩展（主入口）
        
        Args:
            fragments: 文档片段列表
            vectors: 片段向量数组
            doc_ids: 每个片段所属文档ID
            doc_orders: 每个片段在文档中的顺序
            question: 用户问题
            top_k: 初始检索数量
            context_window: 上下文扩展窗口
            max_tokens: 最大token数
            alpha: BM25权重（默认使用实例的alpha）
            
        Returns:
            (context, audit): 上下文文本和审计信息
        """
        use_alpha = alpha if alpha is not None else self.alpha
        
        # 初始检索
        context, audit = self._single_retrieve(
            fragments, vectors, doc_ids, doc_orders, question,
            top_k, context_window, max_tokens, use_alpha
        )

        # 动态扩展：如果上下文不足，扩大检索范围
        min_tokens = int(max_tokens * self.min_context_ratio)
        attempt = 0
        final_top_k = top_k
        final_context_window = context_window

        while audit['total_tokens'] < min_tokens and attempt < self.max_expansion_attempts:
            final_top_k = min(final_top_k + 5, 30)
            final_context_window = min(final_context_window + 1, 5)

            context, audit = self._single_retrieve(
                fragments, vectors, doc_ids, doc_orders, question,
                final_top_k, final_context_window, max_tokens, use_alpha
            )
            attempt += 1

        # 更新审计信息
        audit['final_top_k'] = final_top_k
        audit['final_context_window'] = final_context_window
        audit['expansion_attempts'] = attempt
        audit['bm25_weight'] = use_alpha
        audit['dense_weight'] = 1.0 - use_alpha

        return context, audit


class EnhancedRetrievalPipeline:
    """增强型检索流程：整合去重、重排序、上下文构建"""
    
    def __init__(
        self,
        encoder,
        tokenizer,
        alpha: float = 0.7,
        similarity_threshold: float = 0.85,
        max_context_tokens: int = 4096
    ):
        """
        Args:
            encoder: SentenceTransformer 编码器
            tokenizer: Tokenizer
            alpha: BM25权重
            similarity_threshold: 去重相似度阈值
            max_context_tokens: 最大上下文token数
        """
        self.retriever = HybridRetriever(encoder, tokenizer, alpha)
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.similarity_threshold = similarity_threshold
        self.max_context_tokens = max_context_tokens
        
    def _deduplicate_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """基于语义相似度去重"""
        if len(chunks) <= 1:
            return chunks
        
        texts = [c["content"] for c in chunks]
        embs = self.encoder.encode(texts)
        
        keep_indices = [0]
        for i in range(1, len(embs)):
            sims = cosine_similarity([embs[i]], embs[keep_indices])[0]
            if max(sims) < self.similarity_threshold:
                keep_indices.append(i)
        
        return [chunks[i] for i in keep_indices]
    
    def _rerank_by_relevance(self, query: str, chunks: List[Dict], top_k: int = 5) -> List[Dict]:
        """基于问题-文档相关性重排序"""
        if not chunks:
            return chunks
        
        query_emb = self.encoder.encode([query])[0]
        chunk_texts = [c["content"] for c in chunks]
        chunk_embs = self.encoder.encode(chunk_texts)
        
        similarities = cosine_similarity([query_emb], chunk_embs)[0]
        
        for i, chunk in enumerate(chunks):
            original_score = chunk.get("score", 0)
            rerank_score = float(similarities[i])
            chunk["rerank_score"] = rerank_score
            chunk["combined_score"] = original_score * 0.4 + rerank_score * 0.6
        
        chunks.sort(key=lambda x: x["combined_score"], reverse=True)
        return chunks[:top_k]
    
    def build_context_prompt(self, question: str, context: str) -> str:
        """构建带文档来源标记说明的Prompt"""
        return f"""你是一个严谨的问答助手。请**严格基于**以下上下文信息回答问题。

## 上下文信息说明
上下文中每条信息以 `[文档数字]` 开头，代表它来自不同的文档。
请根据问题，从**最相关的文档**中提取答案。

## 上下文
{context}

## 要求
- 如果上下文中包含与问题相关的信息，请准确、完整地回答。
- 如果上下文中**没有**相关信息，请直接回答："根据所提供的上下文，无法回答该问题。"
- 不要引入上下文以外的知识或进行推测。
- 若存在多个可能答案，请指明最相关的一个，并简要说明理由。

## 问题
{question}

## 答案
"""
    
    def run(
        self,
        fragments: List[str],
        vectors: np.ndarray,
        doc_ids: List[int],
        doc_orders: List[int],
        question: str,
        top_k: int = 5,
        context_window: int = 2,
        alpha: Optional[float] = None
    ) -> Tuple[str, dict]:
        """
        完整检索流程
        
        Returns:
            (prompt, audit): 完整的Prompt和审计信息
        """
        # 1. 混合检索
        context, audit = self.retriever.retrieve(
            fragments, vectors, doc_ids, doc_orders, question,
            top_k, context_window, self.max_context_tokens, alpha
        )
        
        # 2. 构建最终Prompt
        prompt = self.build_context_prompt(question, context)
        
        return prompt, audit
