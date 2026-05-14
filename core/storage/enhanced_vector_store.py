"""
增强版向量数据库 - 支持信息融合和智能检索
"""
import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import os
import pickle


class InfoRecord:
    """信息记录"""
    def __init__(self, content: str, source_id: str = "", metadata: Dict = None):
        self.content = content
        self.source_id = source_id
        self.metadata = metadata or {}
        self.record_id = f"rec_{datetime.now().timestamp()}_{id(self)}"
        self.created_at = datetime.now().timestamp()
        self.updated_at = self.created_at
        self.merged_from = []  # 记录融合的来源ID
        self.mention_count = 1  # 引用次数
        self.importance = 1.0  # 重要性评分
    
    def to_dict(self) -> Dict:
        return {
            'record_id': self.record_id,
            'content': self.content,
            'source_id': self.source_id,
            'metadata': self.metadata,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'merged_from': self.merged_from,
            'mention_count': self.mention_count,
            'importance': self.importance
        }
    
    @staticmethod
    def from_dict(data: Dict) -> 'InfoRecord':
        record = InfoRecord(
            content=data['content'],
            source_id=data.get('source_id', ''),
            metadata=data.get('metadata', {})
        )
        record.record_id = data['record_id']
        record.created_at = data.get('created_at', record.created_at)
        record.updated_at = data.get('updated_at', record.updated_at)
        record.merged_from = data.get('merged_from', [])
        record.mention_count = data.get('mention_count', 1)
        record.importance = data.get('importance', 1.0)
        return record


class EnhancedVectorStore:
    """
    增强版向量数据库
    支持：
    1. 语义相似度检索
    2. 自动信息融合
    3. 持久化存储
    4. 智能去重
    """
    
    def __init__(self, encoder_model, dimension: int = 384, 
                 persist_path: str = None, fusion_threshold: float = 0.8):
        """
        Args:
            encoder_model: SentenceTransformer 编码器
            dimension: 向量维度
            persist_path: 持久化路径
            fusion_threshold: 融合阈值
        """
        self.encoder = encoder_model
        self.dimension = dimension
        self.fusion_threshold = fusion_threshold
        self.persist_path = persist_path
        
        # FAISS 索引（内积）
        self.index = faiss.IndexFlatIP(dimension)
        
        # 信息记录存储
        self.records: List[InfoRecord] = []
        self.record_map: Dict[str, InfoRecord] = {}  # record_id -> InfoRecord
        
        # 加载持久化数据
        if persist_path and os.path.exists(persist_path):
            self.load()
    
    def add(self, content: str, source_id: str = "", 
            metadata: Dict = None, vector: np.ndarray = None,
            auto_fusion: bool = True) -> InfoRecord:
        """
        添加信息记录
        
        Args:
            content: 文本内容
            source_id: 来源标识
            metadata: 元数据
            vector: 预计算的向量（可选）
            auto_fusion: 是否自动融合相似信息
        
        Returns:
            InfoRecord 对象
        """
        # 检查是否有相似记录
        if auto_fusion and len(self.records) > 0:
            similar_records = self._find_similar(content, threshold=self.fusion_threshold)
            
            if similar_records:
                # 找到相似记录，进行融合
                target_record = similar_records[0][0]  # 最相似的记录
                self._merge_records(target_record, content, source_id, metadata)
                return target_record
        
        # 创建新记录
        record = InfoRecord(content=content, source_id=source_id, metadata=metadata)
        
        # 计算向量
        if vector is None:
            vector = self.encoder.encode([content])[0]
        
        # 归一化
        vector = vector / np.linalg.norm(vector)
        
        # 添加到索引
        self.index.add(vector.reshape(1, -1))
        self.records.append(record)
        self.record_map[record.record_id] = record
        
        return record
    
    def _find_similar(self, content: str, threshold: float = 0.8, 
                     top_k: int = 5) -> List[Tuple[InfoRecord, float]]:
        """
        查找相似记录
        
        Returns:
            [(record, similarity_score), ...]
        """
        if not self.records:
            return []
        
        # 编码查询
        query_vec = self.encoder.encode([content])[0]
        query_vec = query_vec / np.linalg.norm(query_vec)
        
        # 检索
        k = min(top_k, len(self.records))
        scores, indices = self.index.search(query_vec.reshape(1, -1), k)
        
        # 过滤低于阈值的结果
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx != -1 and score >= threshold:
                results.append((self.records[idx], float(score)))
        
        return results
    
    def _merge_records(self, target: InfoRecord, new_content: str,
                      new_source_id: str, new_metadata: Dict) -> None:
        """
        融合信息到目标记录
        
        策略：
        1. 保留更完整的内容
        2. 更新元数据
        3. 增加引用次数
        4. 记录融合历史
        """
        # 1. 智能内容融合
        if len(new_content) > len(target.content):
            # 新内容更长，检查是否包含新信息
            if new_content[:100] not in target.content:
                target.content = target.content + "\n\n[补充信息]\n" + new_content
        else:
            # 目标内容更长，检查是否需要补充
            if target.content[:100] not in new_content:
                if new_content[:200] not in target.content:
                    target.content = target.content + "\n\n[补充信息]\n" + new_content
        
        # 2. 更新重要性
        target.mention_count += 1
        target.importance = max(target.importance, 1.0 + target.mention_count * 0.1)
        target.updated_at = datetime.now().timestamp()
        
        # 3. 记录融合历史
        target.merged_from.append(f"{new_source_id}_{datetime.now().timestamp()}")
        
        # 4. 合并来源ID
        if new_source_id and new_source_id not in target.source_id:
            target.source_id = f"{target.source_id}+{new_source_id}" if target.source_id else new_source_id
        
        # 5. 合并元数据
        if new_metadata:
            target.metadata.update(new_metadata)
    
    def search(self, query: str, top_k: int = 5, 
               min_score: float = 0.0) -> List[Tuple[InfoRecord, float]]:
        """
        检索最相似的信息记录
        
        Args:
            query: 查询文本
            top_k: 返回数量
            min_score: 最低相似度阈值
        
        Returns:
            [(record, score), ...]
        """
        if not self.records:
            return []
        
        # 编码查询
        query_vec = self.encoder.encode([query])[0]
        query_vec = query_vec / np.linalg.norm(query_vec)
        
        # 检索
        k = min(top_k, len(self.records))
        scores, indices = self.index.search(query_vec.reshape(1, -1), k)
        
        # 构建结果
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx != -1 and score >= min_score:
                results.append((self.records[idx], float(score)))
        
        return results
    
    def search_and_rank(self, query: str, top_k: int = 5,
                       use_importance: bool = True) -> List[Tuple[InfoRecord, float]]:
        """
        检索并按综合分数排序
        
        综合分数 = 语义相似度 * 0.7 + 归一化重要性 * 0.3
        
        Args:
            query: 查询文本
            top_k: 返回数量
            use_importance: 是否使用重要性加权
        
        Returns:
            [(record, combined_score), ...]
        """
        results = self.search(query, top_k=top_k * 2)  # 多检索一些用于排序
        
        if not use_importance or not results:
            return results[:top_k]
        
        # 计算综合分数
        max_importance = max(r.mention_count * r.importance for r, _ in results)
        
        scored_results = []
        for record, sim_score in results:
            importance_score = (record.mention_count * record.importance) / max_importance
            combined_score = sim_score * 0.7 + importance_score * 0.3
            scored_results.append((record, combined_score))
        
        # 按综合分数排序
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        return scored_results[:top_k]
    
    def get_all_records(self, sort_by_importance: bool = True) -> List[InfoRecord]:
        """获取所有记录"""
        if sort_by_importance:
            return sorted(
                self.records,
                key=lambda r: r.mention_count * r.importance,
                reverse=True
            )
        return self.records.copy()
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        if not self.records:
            return {
                'total_records': 0,
                'total_mentions': 0,
                'avg_importance': 0.0,
                'merged_records': 0
            }
        
        total_mentions = sum(r.mention_count for r in self.records)
        avg_importance = np.mean([r.mention_count * r.importance for r in self.records])
        merged_records = sum(1 for r in self.records if r.merged_from)
        
        return {
            'total_records': len(self.records),
            'total_mentions': total_mentions,
            'avg_importance': float(avg_importance),
            'merged_records': merged_records
        }
    
    def save(self):
        """持久化到文件"""
        if not self.persist_path:
            return
        
        data = {
            'index': faiss.serialize_index(self.index),
            'records': [r.to_dict() for r in self.records],
            'dimension': self.dimension
        }
        
        # 确保目录存在
        os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
        
        with open(self.persist_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"[向量数据库] 已保存到: {self.persist_path}")
    
    def load(self):
        """从文件加载"""
        if not self.persist_path or not os.path.exists(self.persist_path):
            return
        
        with open(self.persist_path, 'rb') as f:
            data = pickle.load(f)
        
        self.index = faiss.deserialize_index(data['index'])
        self.records = [InfoRecord.from_dict(d) for d in data['records']]
        self.record_map = {r.record_id: r for r in self.records}
        
        print(f"[向量数据库] 已加载: {self.persist_path}, 记录数: {len(self.records)}")
    
    def clear(self):
        """清空所有数据"""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.records = []
        self.record_map = {}
        
        # 清空持久化文件
        if self.persist_path and os.path.exists(self.persist_path):
            os.remove(self.persist_path)
            print(f"[向量数据库] 已清空并删除: {self.persist_path}")

