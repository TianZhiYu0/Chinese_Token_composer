# Core 模块结构说明

## 📁 模块组织

核心组件已按功能领域进行逻辑分组：

### 1️⃣ 检索模块 (Retrieval)
负责文档检索和查询优化

| 文件 | 类/函数 | 功能 |
|------|---------|------|
| `query_expander.py` | `QueryExpander` | 查询扩展（同义词替换、实体识别） |
| `result_compressor.py` | `ResultCompressor` | 检索结果去重和长度剪枝 |
| `context_builder.py` | `ContextBuilder` | 智能上下文构建（历史+检索） |
| `indexer.py` | `BM25Indexer` | BM25 关键词索引 |

**使用示例：**

```python
from core.modules import QueryExpander, ResultCompressor

# 或者直接从文件导入
from core.retrieval.query_expander import QueryExpander
```

---

### 2️⃣ 压缩模块 (Compression)
负责文档和提示词压缩

| 文件 | 类/函数 | 功能 |
|------|---------|------|
| `compressor.py` | `HardCompressor` | BERT Token级硬压缩 |
| `merger.py` | `PromptMerger` | 语义相似度合并（SentenceTransformer） |

**使用示例：**
```python
from core.modules import HardCompressor, PromptMerger
```

---

### 3️⃣ 历史管理模块 (History)
负责对话历史管理和多源聚合树

| 文件 | 类/函数 | 功能 |
|------|---------|------|
| `tree_history.py` | `MultiSourceTree`, `InfoNode`, `QANode` | 多源聚合树数据结构 |
| `history_manager.py` | `HistoryManager` | 历史管理器（对外接口） |
| `textrank_summarizer.py` | `TextRankSummarizer` | TextRank 无监督摘要 |

**使用示例：**
```python
from core.modules import HistoryManager, MultiSourceTree

# 创建历史管理器
history_mgr = HistoryManager(
    max_context_tokens=2000,
    encoder_model=encoder
)

# 添加对话轮次
history_mgr.add_turn(
    question="什么是人工智能？",
    answer="人工智能是...",
    retrieved_chunks=[...]
)
```

---

### 4️⃣ 存储模块 (Storage)
负责向量存储和信息管理

| 文件 | 类/函数 | 功能 |
|------|---------|------|
| `vector_store.py` | `VectorStore` | 基础 FAISS 向量数据库 |
| `enhanced_vector_store.py` | `EnhancedVectorStore`, `InfoRecord` | 增强向量数据库（支持融合） |

**使用示例：**
```python
from core.modules import EnhancedVectorStore

# 创建向量数据库
vector_db = EnhancedVectorStore(
    encoder_model=encoder,
    dimension=384,
    persist_path="data/vector_store.pkl",
    fusion_threshold=0.8
)

# 添加信息（自动融合）
record = vector_db.add(
    content="金盘科技是一家...",
    source_id="doc1.txt",
    auto_fusion=True
)
```

---

### 5️⃣ 可视化模块 (Visualization)
负责多源聚合树可视化

| 文件 | 类/函数 | 功能 |
|------|---------|------|
| `tree_visualizer.py` | `TreeVisualizer`, `visualize_tree` | 生成交互式 HTML 图表 |

**使用示例：**
```python
from core.modules import visualize_tree

# 生成可视化
visualize_tree(
    tree=history_mgr.tree,
    output_path="tree_visualization.html",
    max_nodes=100
)
```

---

### 6️⃣ I/O 模块 (Input/Output)
负责文档读取

| 文件 | 类/函数 | 功能 |
|------|---------|------|
| `document_reader.py` | `DocumentReader` | 多格式文档读取（TXT, PDF, DOCX等） |

**使用示例：**
```python
from core.modules import DocumentReader

reader = DocumentReader()
content = reader.read_file("document.pdf")
```

---

### 7️⃣ 模型模块 (Models)
负责 LLM 客户端

| 文件 | 类/函数 | 功能 |
|------|---------|------|
| `llm_client.py` | `LLMClient` | LLM 客户端（支持 Ollama/OpenAI） |
| `llm_clientP.py` | `LLMClientP` | 备用 LLM 客户端 |

**使用示例：**
```python
from core.modules import LLMClient

# Ollama
client = LLMClient(
    backend_type="ollama",
    api_url="http://localhost:11434/api/chat",
    model_name="qwen2.5:3b"
)

# OpenAI 兼容 API
client = LLMClient(
    backend_type="openai",
    api_url="https://api.openai.com/v1/chat/completions",
    model_name="gpt-3.5-turbo",
    api_key="sk-xxx"
)
```

---

### 8️⃣ 工具模块 (Utils)
通用工具函数

| 文件 | 函数 | 功能 |
|------|------|------|
| `utils.py` | `split_sentences` | 句子分割 |
| | `count_tokens` | Token 计数 |
| | `chinese_word_seg` | 中文分词 |

---

### 9️⃣ 核心引擎 (Core Engine)
问答系统主入口

| 文件 | 类/函数 | 功能 |
|------|---------|------|
| `qa_engine.py` | `QAEngine` | 问答引擎（整合所有模块） |
| `summarizer.py` | `DialogSummarizer` | 对话摘要生成器 |

**使用示例：**
```python
from core.modules import QAEngine

# 创建问答引擎
qa_engine = QAEngine(
    llm_client=llm_client,
    tokenizer=tokenizer,
    encoder=encoder,
    summarizer=summarizer,
    config=config,
    vector_store_path="data/vector_store.pkl"
)

# 回答问题
answer, info = qa_engine.answer(
    question="金盘科技的发展如何？",
    documents=documents,
    alpha=0.5,
    use_expansion=True,
    use_compression=True,
    use_smart_context=True
)
```

---

## 🎯 快速导入指南

### 方式 1：使用 modules.py（推荐）
```python
from core.modules import (
    QAEngine,
    HistoryManager,
    EnhancedVectorStore,
    visualize_tree
)
```

### 方式 2：直接从文件导入

```python
from core.engine.qa_engine import QAEngine
from core.history.history_manager import HistoryManager
from core.storage.enhanced_vector_store import EnhancedVectorStore
```

### 方式 3：使用 __init__.py
```python
from core import QAEngine, HistoryManager, visualize_tree
```

---

## 📊 模块依赖关系

```
QAEngine (主入口)
├── 检索模块
│   ├── QueryExpander
│   ├── ResultCompressor
│   └── ContextBuilder
├── 压缩模块
│   └── HardCompressor
├── 历史模块
│   ├── HistoryManager
│   │   ├── MultiSourceTree
│   │   ├── EnhancedVectorStore
│   │   └── TextRankSummarizer
│   └── TreeVisualizer
└── 模型模块
    └── LLMClient
```

---

## 🚀 完整使用示例

```python
import config
from core.modules import (
    QAEngine,
    LLMClient,
    HardCompressor,
    PromptMerger,
    DialogSummarizer,
    DocumentReader,
    visualize_tree
)
from transformers import AutoTokenizer

# 1. 初始化组件
compressor = HardCompressor(config.MODEL_PATH)
tokenizer = compressor.tokenizer
merger = PromptMerger(config.ENCODER_NAME)
summarizer = DialogSummarizer(config.SUMMARIZER_BASE)

# 2. 创建 LLM 客户端
llm_client = LLMClient(
    backend_type="ollama",
    api_url=config.OLLAMA_URL,
    model_name=config.MODEL_NAME,
    tokenizer=tokenizer
)

# 3. 创建问答引擎
qa_engine = QAEngine(
    llm_client=llm_client,
    tokenizer=tokenizer,
    encoder=merger.encoder,
    summarizer=summarizer,
    config=config,
    vector_store_path="data/vector_store.pkl"
)

# 4. 读取文档
reader = DocumentReader()
content = reader.read_file("document.pdf")
documents = [content]

# 5. 回答问题
answer, info = qa_engine.answer(
    question="文档的主要内容是什么？",
    documents=documents,
    alpha=0.5
)

print(f"答案: {answer}")
print(f"检索到的chunks: {len(info['retrieved_chunks'])}")

# 6. 可视化历史树
visualize_tree(
    tree=qa_engine.history_manager.tree,
    output_path="tree_visualization.html"
)
```

---

## 📝 版本信息

- **当前版本**: 2.0.0
- **更新日期**: 2026-04-08
- **维护者**: Prompt-Composer Team
