"""
关键词检索器：基于jieba分词的倒排索引检索
用于从压缩后的片段中快速召回相关片段
"""
import jieba.posseg as pseg
from collections import defaultdict
from typing import List, Set


class KeywordRetriever:
    """基于关键词倒排索引的片段检索器"""
    
    def __init__(self):
        self.inverted_index = defaultdict(set)
        self.fragments = []

    def build_index(self, compressed_fragments: List[str]):
        """
        构建倒排索引
        
        参数:
            compressed_fragments: 压缩后的文本片段列表
        """
        self.fragments = compressed_fragments
        self.inverted_index.clear()
        
        for idx, frag in enumerate(compressed_fragments):
            keywords = self._extract_keywords(frag)
            for kw in keywords:
                self.inverted_index[kw].add(idx)
        
        print(f"✅ 倒排索引构建完成: {len(self.fragments)} 个片段, {len(self.inverted_index)} 个关键词")

    def _extract_keywords(self, text: str) -> Set[str]:
        """
        提取文本中的关键词
        
        参数:
            text: 输入文本
            
        返回:
            关键词集合（名词、地名、人名、数字等）
        """
        words = pseg.cut(text)
        keywords = set()
        
        # 词性过滤：保留名词类、地名、人名、机构名、其他专名、英文、数字
        for w, flag in words:
            if flag in ('n', 'nr', 'ns', 'nt', 'nz', 'eng', 'm') and len(w.strip()) > 1:
                keywords.add(w.strip())
        
        return keywords

    def retrieve(self, question: str, top_k: int = 5) -> List[str]:
        """
        根据问题检索相关片段
        
        参数:
            question: 查询问题
            top_k: 返回前k个最相关的片段
            
        返回:
            检索到的片段列表
        """
        # 提取问题关键词
        q_keywords = self._extract_keywords(question)
        
        if not q_keywords:
            # 如果没有提取到关键词，返回前top_k个片段
            return self.fragments[:top_k]
        
        # 统计每个片段命中的关键词数量
        hit_counts = defaultdict(int)
        for kw in q_keywords:
            for idx in self.inverted_index.get(kw, set()):
                hit_counts[idx] += 1
        
        # 按命中次数降序排序，取top_k
        sorted_indices = sorted(hit_counts.keys(), key=lambda x: hit_counts[x], reverse=True)[:top_k]
        
        # 返回对应片段
        result_fragments = [self.fragments[idx] for idx in sorted_indices]
        
        return result_fragments
