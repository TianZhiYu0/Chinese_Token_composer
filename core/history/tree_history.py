"""
多源聚合树历史管理器
替代原有的线性摘要存储
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class InfoNode:
    """信息节点（第一层）- 来自文档片段"""
    content: str
    source_id: str = ""
    importance: float = 1.0
    children: List['QANode'] = field(default_factory=list)
    node_id: str = field(default_factory=lambda: f"info_{datetime.now().timestamp()}_{id(None)}")
    
    # 新增：融合相关字段
    merged_from: List[str] = field(default_factory=list)  # 记录融合的来源节点ID
    last_updated: float = field(default_factory=lambda: datetime.now().timestamp())  # 最后更新时间
    mention_count: int = 1  # 被引用次数


@dataclass
class QANode:
    """问答节点 - 问题和答案对"""
    question: str
    answer: str = ""
    parents: List[Any] = field(default_factory=list)  # 可以是 InfoNode 或 QANode
    children: List['QANode'] = field(default_factory=list)
    timestamp: int = 0
    node_id: str = field(default_factory=lambda: f"qa_{datetime.now().timestamp()}_{id(None)}")


class MultiSourceTree:
    """多源聚合树"""

    def __init__(self):
        self.root = None  # 空根节点，不携带任何信息
        self.info_nodes: List[InfoNode] = []
        self.qa_nodes: List[QANode] = []
        self._next_timestamp = 0

    def add_info(self, content: str, source_id: str = "") -> InfoNode:
        """添加信息源节点"""
        node = InfoNode(content=content, source_id=source_id)
        self.info_nodes.append(node)
        return node

    def add_qa(self, question: str, answer: str,
               source_infos: Optional[List[InfoNode]] = None,
               parent_qas: Optional[List[QANode]] = None) -> QANode:
        """
        添加一轮问答

        Args:
            question: 用户问题
            answer: 系统答案
            source_infos: 依赖的信息节点（文档片段）
            parent_qas: 依赖的父问答节点（用于递归问答）
        """
        node = QANode(
            question=question,
            answer=answer,
            timestamp=self._next_timestamp
        )
        self._next_timestamp += 1

        # 关联信息节点
        if source_infos:
            for info in source_infos:
                node.parents.append(info)
                info.children.append(node)

        # 关联父问答节点
        if parent_qas:
            for parent in parent_qas:
                node.parents.append(parent)
                parent.children.append(node)

        # 如果没有指定任何依赖，直接挂载到根
        if not source_infos and not parent_qas:
            node.parents.append(self.root)

        self.qa_nodes.append(node)
        return node

    def find_info_by_content(self, content: str, threshold: float = 0.8) -> Optional[InfoNode]:
        """根据内容查找已存在的信息节点（用于去重）"""
        from sentence_transformers import SentenceTransformer
        import os
        
        if not hasattr(self, '_encoder'):
            # 使用本地模型路径
            from config import ENCODER_NAME
            model_path = ENCODER_NAME
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型路径不存在: {model_path}")
            
            # 强制使用本地模型
            self._encoder = SentenceTransformer(
                model_path,
                local_files_only=True
            )

        if not self.info_nodes:
            return None

        emb_query = self._encoder.encode([content])[0]
        emb_existing = self._encoder.encode([n.content for n in self.info_nodes])

        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        sims = cosine_similarity([emb_query], emb_existing)[0]
        best_idx = np.argmax(sims)
        if sims[best_idx] >= threshold:
            return self.info_nodes[best_idx]
        return None

    def get_recent_qa(self, n: int = 5) -> List[QANode]:
        """获取最近的 n 轮问答"""
        return self.qa_nodes[-n:] if self.qa_nodes else []

    def to_context_text(self, max_tokens: int = 2000) -> str:
        """
        将树转换为 LLM 可用的线性上下文
        策略：取最近 5 轮问答 + 关键信息节点摘要
        """
        # 使用新的融合版本
        return self.get_merged_context_text(max_tokens)

    def clear(self):
        """清空树"""
        self.info_nodes.clear()
        self.qa_nodes.clear()
        self._next_timestamp = 0
    
    def merge_similar_info_nodes(self, threshold: float = 0.75) -> int:
        """
        合并相似的信息节点，实现信息融合
        
        Args:
            threshold: 相似度阈值，高于此值则合并
            
        Returns:
            合并的节点对数量
        """
        if len(self.info_nodes) < 2:
            return 0
        
        # 确保编码器已加载
        if not hasattr(self, '_encoder'):
            from sentence_transformers import SentenceTransformer
            import os
            from config import ENCODER_NAME
            
            if not os.path.exists(ENCODER_NAME):
                raise FileNotFoundError(f"模型路径不存在: {ENCODER_NAME}")
            
            self._encoder = SentenceTransformer(
                ENCODER_NAME,
                local_files_only=True
            )
        
        # 计算所有节点的嵌入向量
        contents = [node.content for node in self.info_nodes]
        embeddings = self._encoder.encode(contents)
        
        # 计算相似度矩阵
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        sim_matrix = cosine_similarity(embeddings)
        np.fill_diagonal(sim_matrix, 0)  # 排除自己与自己比较
        
        merged_count = 0
        merged_indices = set()
        
        # 找出所有相似的节点对
        for i in range(len(self.info_nodes)):
            if i in merged_indices:
                continue
            
            for j in range(i + 1, len(self.info_nodes)):
                if j in merged_indices:
                    continue
                
                if sim_matrix[i][j] >= threshold:
                    # 合并节点 j 到节点 i
                    self._merge_two_nodes(self.info_nodes[i], self.info_nodes[j])
                    merged_indices.add(j)
                    merged_count += 1
        
        # 移除已合并的节点
        if merged_indices:
            self.info_nodes = [
                node for idx, node in enumerate(self.info_nodes)
                if idx not in merged_indices
            ]
        
        return merged_count
    
    def _merge_two_nodes(self, target: InfoNode, source: InfoNode) -> None:
        """
        融合两个信息节点
        
        策略：
        1. 保留更多内容
        2. 更新重要性（累加引用次数）
        3. 合并子节点关系
        4. 记录融合历史
        """
        # 1. 智能内容融合：选择更长的内容，或拼接补充信息
        if len(source.content) > len(target.content):
            # 如果源节点内容更长，检查是否包含新信息
            if source.content[:100] not in target.content:
                # 内容不同，进行拼接融合
                target.content = target.content + "\n\n[补充信息]\n" + source.content
            else:
                # 源节点内容已包含在目标中，无需操作
                pass
        else:
            # 目标节点内容更长，检查是否需要补充
            if target.content[:100] not in source.content:
                if source.content[:200] not in target.content:
                    target.content = target.content + "\n\n[补充信息]\n" + source.content
        
        # 2. 更新重要性（基于引用次数）
        target.importance = max(target.importance, source.importance)
        target.mention_count += source.mention_count
        
        # 3. 合并子节点（问答节点）
        for child in source.children:
            if child not in target.children:
                target.children.append(child)
                # 更新子节点的父节点引用
                if target in child.parents:
                    child.parents.remove(source)
        
        # 4. 记录融合历史
        target.merged_from.append(source.node_id)
        target.last_updated = datetime.now().timestamp()
        
        # 5. 合并来源ID
        if source.source_id and source.source_id not in target.source_id:
            target.source_id = f"{target.source_id}+{source.source_id}" if target.source_id else source.source_id
    
    def get_merged_context_text(self, max_tokens: int = 2000, 
                                use_summary: bool = True) -> str:
        """
        获取融合后的上下文文本（优化版）
        
        Args:
            max_tokens: 最大token数
            use_summary: 是否使用摘要模式
            
        Returns:
            格式化后的上下文文本
        """
        context_parts = []
        
        # 1. 最近的问答
        recent_qa = self.get_recent_qa(5)
        if recent_qa:
            context_parts.append("【对话历史】")
            for qa in recent_qa:
                context_parts.append(f"用户：{qa.question}")
                context_parts.append(f"系统：{qa.answer[:500]}")
        
        # 2. 关键信息节点（按重要性和引用次数排序）
        if self.info_nodes:
            # 排序策略：引用次数 * 重要性
            scored_infos = sorted(
                self.info_nodes,
                key=lambda x: x.mention_count * x.importance,
                reverse=True
            )[:5]  # 取top 5
            
            context_parts.append("\n【关键信息来源】")
            for info in scored_infos:
                # 显示融合标记
                merge_tag = f" [融合{len(info.merged_from)}个源]" if info.merged_from else ""
                content_preview = info.content[:200]
                context_parts.append(
                    f"- {content_preview}{merge_tag}\n"
                    f"  (引用{info.mention_count}次, 来源: {info.source_id})"
                )
        
        return "\n".join(context_parts)