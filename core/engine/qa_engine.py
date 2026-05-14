"""
智能问答引擎 - 支持混合检索、上下文压缩、历史摘要管理
整合优化模块：查询扩展、检索结果去重剪枝、智能上下文构建
"""
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import normalize
from typing import List, Dict, Any, Tuple
import jieba

# 导入优化模块
from core.retrieval.query_expander import QueryExpander
from core.retrieval.result_compressor import ResultCompressor
from core.retrieval.context_builder import ContextBuilder
from core.retrieval.dynamic_allocator import DynamicTokenAllocator, SmartDocumentExtractor
from core.retrieval.t5_fusion import T5FusionEngine, HybridResultCompressor
from core.history.history_manager import HistoryManager

# 中文分词函数
def chinese_tokenizer(text: str) -> List[str]:
    return list(jieba.cut(text))


class QAEngine:
    """问答引擎，整合检索、压缩、生成与历史管理"""

    def __init__(self, llm_client, tokenizer, encoder, summarizer, config,
                 vector_store_path: str = None, use_t5_fusion: bool = False,
                 use_dynamic_allocation: bool = True):
        """
        Args:
            use_dynamic_allocation: 是否启用动态token分配
        """
        self.llm_client = llm_client
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.summarizer = summarizer
        self.config = config

        # 初始化优化组件（可配置开关）
        self.query_expander = QueryExpander()
        
        # 可选：使用 T5 融合去重
        if use_t5_fusion:
            print("🔄 初始化 T5 融合引擎...")
            self.t5_fusion = T5FusionEngine(
                model_path=config.CHUNK_FUSER_MODEL,  # 使用配置文件中的路径（支持 LoRA）
                device=config.DEVICE  # 使用配置中的设备（自动检测 GPU）
            )
            self.result_compressor = HybridResultCompressor(
                encoder_model=encoder,
                t5_fusion_engine=self.t5_fusion,
                use_t5_fusion=True
            )
            print("✅ T5 融合去重已启用（LoRA Adapter）")
        else:
            self.t5_fusion = None
            self.result_compressor = ResultCompressor(encoder)
        # 动态token分配器
        self.use_dynamic_allocation = True  # 默认启用
        self.token_allocator = DynamicTokenAllocator(max_total_tokens=2500)
        
        self.context_builder = ContextBuilder(tokenizer, max_tokens=config.CONTEXT_WINDOW_SIZE)
        
        # 初始化多源聚合树历史管理器（带向量数据库）
        self.history_manager = HistoryManager(
            max_context_tokens=config.MAX_CONTEXT_TOKENS,
            encoder_model=encoder,
            vector_store_path=vector_store_path
        )

    def answer_with_strategy(self, question: str, strategy_mode: str,
                             compressed_full_text: str = None,
                             fragments: List[str] = None,
                             keyword_retriever=None) -> Tuple[str, Dict]:
        if strategy_mode == "full_compress":
            return self.answer_direct(question, compressed_full_text)
        else:
            # 根据是否有 keyword_retriever 选择检索方式
            if keyword_retriever:
                retrieved = keyword_retriever.retrieve(question, top_k=config.KEYWORD_RETRIEVAL_TOP_K)
                context = "\n\n".join(retrieved)
                return self.answer_direct(question, context)
            else:
                return self.answer(question, fragments, ...)  # 原有混合检索

    def answer_direct(self, question: str, context: str) -> Tuple[str, Dict]:
        prompt = f"请基于以下文档内容回答问题。\n\n【文档】\n{context}\n\n【问题】\n{question}\n\n【答案】"
        answer, prompt_tokens, answer_tokens = self.llm_client.call(prompt)
        info = {
            "token_stats": {
                "prompt_tokens": prompt_tokens,
                "answer_tokens": answer_tokens,
                "total": prompt_tokens + answer_tokens
            },
            "retrieved_chunks": [],
            "mode": "full_compress_direct"
        }
        return answer, info

    def answer(
        self,
        question: str,
        documents: List[str],
        alpha: float = 0.0,
        # 优化开关
        use_expansion: bool = True,
        use_compression: bool = True,
        use_smart_context: bool = True,
        compression_ratio: float = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        执行问答
        参数:
            compression_ratio: 压缩比（0.0-1.0），例如0.7表示保留70%的token
                              如果为None，则使用compressor默认策略
        返回: (answer, info_dict)  info_dict 包含 token_stats, retrieved_chunks 等
        """
        # 0. 从多源聚合树自动获取历史上下文
        history_context = self.history_manager.get_context_for_llm(
            max_tokens=int(self.context_builder.max_tokens * 0.3)  # 历史占30%
        )
        # 1. 查询扩展
        queries = [question]
        if use_expansion:
            queries = self.query_expander.expand(question, max_expansions=3)

        # 2. 构建 BM25 索引（基于文档片段）
        tokenized_docs = [chinese_tokenizer(doc) for doc in documents]
        bm25 = BM25Okapi(tokenized_docs)

        # 3. 对每个查询检索并合并结果
        all_chunks = []  # 每个元素: {"content": str, "score": float, "index": int}
        seen_contents = set()

        for q in queries:
            # BM25 分数
            bm25_scores = bm25.get_scores(chinese_tokenizer(q))
            # 向量相似度
            q_vec = self.encoder.encode([q])[0]
            q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-8)
            doc_vectors = np.array([self.encoder.encode([doc])[0] for doc in documents])
            doc_vectors = normalize(doc_vectors, norm='l2')
            vec_scores = np.dot(doc_vectors, q_vec)

            # 混合分数：alpha 映射到 [0,1]
            alpha_clip = max(0.0, min(1.0, (alpha + 1) / 2))
            # 归一化 BM25
            bm25_min, bm25_max = np.min(bm25_scores), np.max(bm25_scores)
            if bm25_max - bm25_min > 1e-8:
                bm25_norm = (bm25_scores - bm25_min) / (bm25_max - bm25_min)
            else:
                bm25_norm = np.zeros_like(bm25_scores)
            final_scores = alpha_clip * vec_scores + (1 - alpha_clip) * bm25_norm

            # 取 top-10
            top_indices = np.argsort(final_scores)[-10:][::-1]
            for idx in top_indices:
                content = documents[idx]
                if content not in seen_contents:
                    seen_contents.add(content)
                    all_chunks.append({
                        "content": content,
                        "score": float(final_scores[idx]),
                        "index": idx
                    })

        # 按分数排序
        all_chunks.sort(key=lambda x: x["score"], reverse=True)

        # 4. 检索结果压缩（去重 + 长度剪枝 + 智能提取）
        if use_compression and all_chunks:
            # 关键优化1：提高去重阈值，减少冗余chunks
            all_chunks = self.result_compressor.deduplicate(all_chunks, threshold=0.92)
            
            # 关键优化2：限制最大chunks数量
            max_chunks = 3  # 严格限制为3个
            if len(all_chunks) > max_chunks:
                all_chunks = all_chunks[:max_chunks]
            
            # 动态Token分配
            if self.use_dynamic_allocation:
                # 计算可用文档tokens
                available_tokens = int(self.context_builder.max_tokens * 0.5)
                allocation = self.token_allocator.allocate_tokens(question, available_tokens)
                
                # 智能提取关键信息
                all_chunks = SmartDocumentExtractor.extract_by_importance(
                    all_chunks, 
                    max_total_chars=allocation["retrieval_tokens"] * 2  # 字符约为token的2倍
                )
                
                max_chars = allocation["retrieval_tokens"] * 2
                all_chunks = self.result_compressor.prune_by_length(all_chunks, max_total_chars=max_chars)
            else:
                # 原始逻辑
                max_chars = int(self.context_builder.max_tokens * 0.4)  # 减少到40%
                all_chunks = self.result_compressor.prune_by_length(all_chunks, max_total_chars=max_chars)

        # 5. 构建最终上下文
        if use_smart_context:
            # 减少检索内容占比，从0.7降到0.5
            final_context = self.context_builder.build(all_chunks, history_context, retrieval_ratio=0.5)
        else:
            doc_text = "\n\n".join([c["content"] for c in all_chunks])
            final_context = f"【相关文档】\n{doc_text}\n\n【对话历史】\n{history_context}"

        # 6. 调用 LLM（要求简洁回答）
        prompt = f"""请基于以下信息简洁回答问题（控制在150字以内）。

{final_context}

问：{question}
答："""
        try:
            answer, prompt_tokens, answer_tokens = self.llm_client.call(prompt)
        except Exception as e:
            answer = f"生成答案时出错: {str(e)}"
            prompt_tokens = 0
            answer_tokens = 0

        # 7. Token 统计（使用 call 方法返回的值）
        token_stats = {
            "prompt_tokens": prompt_tokens,
            "answer_tokens": answer_tokens,
            "total": prompt_tokens + answer_tokens
        }
        
        # 8. 自动记录到多源聚合树
        if all_chunks:
            # 有文档检索：传入完整的chunks字典列表（包含content和score）
            self.history_manager.add_turn(
                question=question,
                answer=answer,
                retrieved_chunks=all_chunks  # 直接传入字典列表
            )
        else:
            # 无文档检索：直接记录问答
            self.history_manager.add_turn(question=question, answer=answer)

        # 9. 返回答案和附加信息
        info = {
            "token_stats": token_stats,
            "retrieved_chunks": all_chunks,
            "used_queries": queries,
            "history_tree": self.history_manager.get_tree()  # 返回树结构用于可视化
        }
        return answer, info
