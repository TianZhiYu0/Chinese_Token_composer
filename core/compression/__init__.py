"""
压缩模块 - BERT硬压缩、语义合并和基于注意力的软合并
"""
from .compressor import HardCompressor
from .merger import PromptMerger
from .attention_merger import AttentionBasedMerger, SoftCompressorWithAdapter
from .keyword_retriever import KeywordRetriever
from .preprocessors import DocumentPreprocessor
from .compression_strategy import CompressionStrategy
from .hybrid_compressor import HybridCompressor
from .word_priority_compressor import WordPriorityCompressor

__all__ = [
    'HardCompressor',
    'PromptMerger',
    'DocumentPreprocessor',
    'KeywordRetriever',
    'CompressionStrategy',
    'HybridCompressor',
    'WordPriorityCompressor',
]