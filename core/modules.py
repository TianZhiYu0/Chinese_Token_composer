"""
Core 模块分组 - 逻辑组织
========================

由于文件已在 core/ 根目录，本文件提供逻辑分组和快捷导入。

📦 模块分组
-----------
"""

# ============================================================
# 1. 检索模块 (Retrieval)
# ============================================================
from core.retrieval.query_expander import QueryExpander
from core.retrieval.result_compressor import ResultCompressor
from core.retrieval.context_builder import ContextBuilder
from core.retrieval.indexer import Indexer

# ============================================================
# 2. 压缩模块 (Compression)
# ============================================================
from core.compression.compressor import HardCompressor
from core.compression.merger import PromptMerger

# ============================================================
# 3. 历史管理模块 (History)
# ============================================================
from core.history.tree_history import MultiSourceTree, InfoNode, QANode
from core.history.history_manager import HistoryManager
from core.history.textrank_summarizer import TextRankSummarizer

# ============================================================
# 4. 存储模块 (Storage)
# ============================================================
from core.storage.vector_store import VectorStore
from core.storage.enhanced_vector_store import EnhancedVectorStore, InfoRecord

# ============================================================
# 5. 可视化模块 (Visualization)
# ============================================================
from core.visualization.tree_visualizer import TreeVisualizer, visualize_tree

# ============================================================
# 6. I/O 模块 (Input/Output)
# ============================================================
from core.io.document_reader import DocumentReader

# ============================================================
# 7. 模型模块 (Models)
# ============================================================
from core.model.llm_client import LLMClient
# from .llm_clientP import LLMClientP  # 备用

# ============================================================
# 8. 工具模块 (Utils)
# ============================================================
from core.utils.utils import split_sentences, count_tokens, chinese_word_seg

# ============================================================
# 9. 核心引擎 (Core Engine)
# ============================================================
from core.engine.qa_engine import QAEngine
from core.engine.summarizer import DialogSummarizer

# ============================================================
# 公共导出
# ============================================================
__all__ = [
    # 检索
    'QueryExpander',
    'ResultCompressor',
    'ContextBuilder',
    'Indexer',
    
    # 压缩
    'HardCompressor',
    'PromptMerger',
    
    # 历史
    'MultiSourceTree',
    'InfoNode',
    'QANode',
    'HistoryManager',
    'TextRankSummarizer',
    
    # 存储
    'VectorStore',
    'EnhancedVectorStore',
    'InfoRecord',
    
    # 可视化
    'TreeVisualizer',
    'visualize_tree',
    
    # I/O
    'DocumentReader',
    
    # 模型
    'LLMClient',
    
    # 工具
    'split_sentences',
    'count_tokens',
    'chinese_word_seg',
    
    # 核心
    'QAEngine',
    'DialogSummarizer',
]


# ============================================================
# 模块分组字典（用于动态导入）
# ============================================================
MODULES = {
    'retrieval': [
        'QueryExpander',
        'ResultCompressor',
        'ContextBuilder',
        'Indexer',
    ],
    'compression': [
        'HardCompressor',
        'PromptMerger',
    ],
    'history': [
        'MultiSourceTree',
        'InfoNode',
        'QANode',
        'HistoryManager',
        'TextRankSummarizer',
    ],
    'storage': [
        'VectorStore',
        'EnhancedVectorStore',
        'InfoRecord',
    ],
    'visualization': [
        'TreeVisualizer',
        'visualize_tree',
    ],
    'io': [
        'DocumentReader',
    ],
    'models': [
        'LLMClient',
    ],
    'utils': [
        'split_sentences',
        'count_tokens',
        'chinese_word_seg',
    ],
    'core': [
        'QAEngine',
        'DialogSummarizer',
    ],
}


def get_module_info():
    """获取模块信息"""
    return {
        'version': '2.0.0',
        'modules': MODULES,
        'total_classes': len(__all__),
    }
