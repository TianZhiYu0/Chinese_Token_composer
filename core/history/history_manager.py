"""
历史管理器 - 使用多源聚合树 + TextRank摘要 + 增强向量数据库
"""
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import os
from core.history.tree_history import MultiSourceTree, QANode
from core.history.textrank_summarizer import TextRankSummarizer
from core.storage.enhanced_vector_store import EnhancedVectorStore, InfoRecord

class HistoryManager:
    def __init__(self, max_context_tokens: int = 2000, encoder_model=None,
                 vector_store_path: str = None):
        self.max_tokens = max_context_tokens
        self.tree = MultiSourceTree()
        self._fallback_history: List[Dict] = []

        # 初始化TextRank摘要器（若提供编码器）
        self.summarizer = None
        if encoder_model:
            self.summarizer = TextRankSummarizer(encoder_model)
        
        # 初始化增强向量数据库
        self.vector_store = None
        if encoder_model:
            self.vector_store = EnhancedVectorStore(
                encoder_model=encoder_model,
                dimension=384,  # paraphrase-multilingual-MiniLM-L12-v2 的维度
                persist_path=vector_store_path,
                fusion_threshold=0.8
            )

    def add_turn(self, question: str, answer: str,
                 retrieved_chunks: Optional[List[Dict]] = None,
                 parent_qas: Optional[List[QANode]] = None,
                 auto_merge: bool = True) -> None:
        # 创建信息节点（去重逻辑保留）
        info_nodes = []
        vector_records = []
        
        if retrieved_chunks:
            for chunk in retrieved_chunks[:5]:
                content = chunk.get("content", "")
                source = chunk.get("source", "")
                if not content:
                    continue
                
                existing = self.tree.find_info_by_content(content, threshold=0.85)
                if existing:
                    # 找到相似节点，增加引用次数
                    existing.mention_count += 1
                    existing.last_updated = datetime.now().timestamp()
                    info_nodes.append(existing)
                else:
                    new_info = self.tree.add_info(content=content, source_id=source)
                    info_nodes.append(new_info)
                
                # 同时存储到向量数据库（自动融合）
                if self.vector_store:
                    record = self.vector_store.add(
                        content=content,
                        source_id=source,
                        metadata={
                            'question': question,
                            'timestamp': datetime.now().isoformat()
                        },
                        auto_fusion=True
                    )
                    vector_records.append(record)

        self.tree.add_qa(question, answer,
                         source_infos=info_nodes if info_nodes else None,
                         parent_qas=parent_qas)
        
        # 自动合并相似节点（当信息节点 > 10 时触发）
        if auto_merge and len(self.tree.info_nodes) > 10:
            merged = self.tree.merge_similar_info_nodes(threshold=0.75)
            if merged > 0:
                print(f"[历史管理] 自动合并了 {merged} 个相似信息节点")
        
        # 输出向量数据库统计
        if self.vector_store and vector_records:
            stats = self.vector_store.get_stats()
            print(f"[向量数据库] 总记录: {stats['total_records']}, "
                  f"融合记录: {stats['merged_records']}, "
                  f"总引用: {stats['total_mentions']}")

        # fallback存储
        self._fallback_history.append({"question": question, "answer": answer[:500]})
        if len(self._fallback_history) > 10:
            self._fallback_history.pop(0)

    def get_context_for_llm(self, max_tokens: Optional[int] = None) -> str:
        """使用TextRank生成历史摘要，优化：支持更多轮次"""
        use_tokens = max_tokens or self.max_tokens

        # 优先使用树结构
        if self.tree.qa_nodes:
            # 优化：提取更多最近对话（5→8轮）
            recent_qa = self.tree.get_recent_qa(8)
            if recent_qa:
                history_text = "\n".join([f"用户：{qa.question}\n系统：{qa.answer}" for qa in recent_qa])
                
                # 优化：提高摘要触发阈值（300→500字符）
                if self.summarizer and len(history_text) > 500:
                    # 使用TextRank生成摘要
                    summary = self.summarizer.summarize(history_text)
                    return f"【对话历史摘要】\n{summary}"
                else:
                    # 优化：返回最近3轮完整对话（2→3）
                    return "\n\n".join([f"用户：{qa.question}\n系统：{qa.answer[:200]}" for qa in recent_qa[-3:]])
        # fallback
        return self.get_fallback_text()

    def get_recent_history(self, n: int = 3) -> List[Dict]:
        recent = self.tree.get_recent_qa(n)
        return [{"question": qa.question, "answer": qa.answer} for qa in recent]

    def get_tree(self) -> MultiSourceTree:
        return self.tree

    def clear(self):
        self.tree.clear()
        self._fallback_history.clear()

    def get_fallback_text(self) -> str:
        parts = []
        for h in self._fallback_history[-5:]:
            parts.append(f"用户：{h['question']}\n系统：{h['answer']}")
        return "\n\n".join(parts)
    
    def search_vector_store(self, query: str, top_k: int = 5,
                           use_importance: bool = True) -> List[Tuple[InfoRecord, float]]:
        """
        从向量数据库检索信息
        
        Args:
            query: 查询文本
            top_k: 返回数量
            use_importance: 是否使用重要性加权
        
        Returns:
            [(record, score), ...]
        """
        if not self.vector_store:
            return []
        
        if use_importance:
            return self.vector_store.search_and_rank(query, top_k=top_k)
        else:
            return self.vector_store.search(query, top_k=top_k)
    
    def get_vector_stats(self) -> Dict:
        """获取向量数据库统计信息"""
        if not self.vector_store:
            return {'enabled': False}
        
        stats = self.vector_store.get_stats()
        stats['enabled'] = True
        return stats
    
    def export_vector_data(self, output_path: str) -> None:
        """
        导出向量数据库的所有记录
        
        Args:
            output_path: 输出文件路径（JSON格式）
        """
        import json
        
        if not self.vector_store:
            print("[警告] 向量数据库未初始化")
            return
        
        records = self.vector_store.get_all_records(sort_by_importance=True)
        export_data = {
            'total_records': len(records),
            'export_time': datetime.now().isoformat(),
            'records': [r.to_dict() for r in records]
        }
        
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        print(f"[向量数据库] 已导出 {len(records)} 条记录到: {output_path}")