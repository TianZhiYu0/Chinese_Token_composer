"""
历史管理模块 - 多源聚合树、历史管理器、TextRank摘要
"""
from .tree_history import MultiSourceTree, InfoNode, QANode
from .history_manager import HistoryManager
from .textrank_summarizer import TextRankSummarizer

__all__ = [
    'MultiSourceTree',
    'InfoNode',
    'QANode',
    'HistoryManager',
    'TextRankSummarizer'
]
