"""
Core 模块组织结构
==================

本目录包含多文档智能问答系统的核心组件，按功能领域组织：

📁 模块结构
-----------

retrieval/          # 检索相关组件
├── query_expander.py      - 查询扩展（同义词、实体识别）
├── result_compressor.py   - 检索结果压缩（去重、剪枝）
├── context_builder.py     - 智能上下文构建
└── indexer.py             - BM25 索引器

compression/        # 压缩相关组件  
├── compressor.py          - BERT 硬压缩（Token级别）
└── merger.py              - 语义合并（SentenceTransformer）

history/            # 历史管理组件
├── tree_history.py        - 多源聚合树数据结构
├── history_manager.py     - 历史管理器（对外接口）
└── textrank_summarizer.py - TextRank 摘要生成

storage/            # 存储组件
├── vector_store.py        - 基础向量数据库（FAISS）
└── enhanced_vector_store.py - 增强向量数据库（支持融合）

visualization/      # 可视化组件
└── tree_visualizer.py     - 多源聚合树可视化

io/                 # 输入输出组件
└── document_reader.py     - 多格式文档读取器

models/             # 模型客户端
├── llm_client.py          - LLM 客户端（Ollama/OpenAI）
└── llm_clientP.py         - 备用 LLM 客户端

utils/              # 工具函数
└── utils.py               - 通用工具函数（分词、Token计数）

🎯 主要入口
-----------

qa_engine.py        - 问答引擎（主入口，整合所有模块）
summarizer.py       - 对话摘要生成器


📝 使用示例
-----------

```python
# 方式1：直接从子模块导入
from core.retrieval import QueryExpander
from core.history import HistoryManager
from core.storage import EnhancedVectorStore

# 方式2：从主模块导入
from core.qa_engine import QAEngine
from core.summarizer import DialogSummarizer

# 方式3：使用完整路径
from core.compression.compressor import HardCompressor
from core.visualization.tree_visualizer import visualize_tree
```


🔄 模块依赖关系
--------------

QAEngine (主入口)
├── retrieval (检索)
│   ├── QueryExpander (查询扩展)
│   ├── ResultCompressor (结果压缩)
│   └── ContextBuilder (上下文构建)
├── compression (压缩)
│   └── HardCompressor (BERT压缩)
├── history (历史)
│   ├── HistoryManager (历史管理)
│   │   ├── MultiSourceTree (多源聚合树)
│   │   ├── EnhancedVectorStore (向量数据库)
│   │   └── TextRankSummarizer (摘要)
│   └── TreeVisualizer (可视化)
└── models (模型)
    └── LLMClient (LLM调用)


⚡ 快速开始
-----------

1. 基础使用：
   from core.qa_engine import QAEngine
   
2. 历史管理：
   from core.history import HistoryManager
   
3. 向量存储：
   from core.storage import EnhancedVectorStore
   
4. 可视化：
   from core.visualization import visualize_tree
"""

# 版本信息
__version__ = "2.0.0"
__author__ = "Prompt-Composer Team"

# 核心模块快捷导入
from core.engine.qa_engine import QAEngine
from core.engine.summarizer import DialogSummarizer
from core.history.history_manager import HistoryManager
from core.storage.enhanced_vector_store import EnhancedVectorStore
from core.visualization.tree_visualizer import visualize_tree
from core.io.document_reader import DocumentReader

__all__ = [
    'QAEngine',
    'DialogSummarizer',
    'HistoryManager',
    'EnhancedVectorStore',
    'visualize_tree',
    'DocumentReader'
]
