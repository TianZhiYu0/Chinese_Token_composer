# Prompt-Composer 系统架构文档

> 更新时间：2026-05-03
> 描述：多文档智能问答系统，支持文档压缩、混合检索、T5语义融合、对话历史管理

---

## 📐 系统架构总览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            用户交互层 (Gradio App)                            │
│          app.py | 文件上传 | 问题输入 | 参数调节 | 模式切换 | 答案展示              │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │
┌────────────────────────────────────▼────────────────────────────────────────┐
│                           业务逻辑层 (QAEngine)                               │
│                                                                             │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│   │QueryExpander│  │DynamicAlloc │  │ T5Fusion    │  │ContextBuilder│       │
│   │   查询扩展    │  │ 动态Token分配│  │  语义融合     │  │  上下文构建   │        │
│   └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │                     HistoryManager (对话历史管理)                  │       │
│   │   MultiSourceTree | TextRankSummarizer | EnhancedVectorStore    │       │
│   └─────────────────────────────────────────────────────────────────┘       │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │
┌────────────────────────────────────▼────────────────────────────────────────┐
│                              核心算法层 (Core)                                │
│                                                                             │
│  ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────┐  │
│  │    压缩模块           │  │    检索模块             │  │    存储模块       │  │
│  │    Compression       │  │    Retrieval         │  │    Storage       │  │
│  ├──────────────────────┤  ├──────────────────────┤  ├──────────────────┤  │
│  │ HardCompressor       │  │ Indexer (FAISS)      │  │ VectorStore      │  │
│  │ (MOOSComp/LLMLingua2)│  │ BM25 倒排索引          │  │ EnhancedVecStore │  │
│  │                      │  │                      │  │                  │  │
│  │ PromptMerger         │  │ QueryExpander        │  │                  │  │
│  │ (语义合并)             │  │ (同义词/实体识别)       │  │                  │  │
│  │                      │  │                      │  │                  │  │
│  │ DocumentPreprocessor │  │ ResultCompressor     │  │                  │  │
│  │ (6种预处理策略)         │  │ (去重/剪枝)           │  │                  │  │
│  │                      │  │                      │  │                  │  │
│  │ WordPriorityCompressor│  │ T5FusionEngine      │  │                  │  │
│  │ (词级优先级)           │  │ (LoRA微调T5)          │  │                  │  │
│  └──────────────────────┘  └──────────────────────┘  └──────────────────┘  │
│                                                                             │
│  ┌──────────────────────┐  ┌──────────────────────┐                        │
│  │    历史模块            │  │    引擎模块           │                        │
│  │    History           │  │    Engine            │                        │
│  ├──────────────────────┤  ├──────────────────────┤                        │
│  │ HistoryManager       │  │ QAEngine             │                        │
│  │ MultiSourceTree      │  │ DialogSummarizer     │                        │
│  │ TextRankSummarizer   │  │                      │                        │
│  └──────────────────────┘  └──────────────────────┘                        │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │
┌────────────────────────────────────▼────────────────────────────────────────┐
│                            模型与数据层 (Models)                               │ 
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │  BERT压缩模型     │  │  T5融合模型      │  │  编码器模型       │              │
│  ├─────────────────┤  ├─────────────────┤  ├─────────────────┤              │
│  │ compression_    │  │ mengzi-t5-      │  │ paraphrase-     │              │
│  │ bert_mooscomp   │  │ finetuned-      │  │ multilingual-   │              │
│  │ _news           │  │ fusion (LoRA)   │  │ MiniLM-L12-v2   │              │
│  │                 │  │                 │  │                 │              │
│  │ llmlingua-2-    │  │                 │  │ sentence_       │              │
│  │ bert-base-      │  │                 │  │ labeling_       │              │
│  │ multilingual-   │  │                 │  │ adapter_cosine  │              │
│  │ cased-          │  │                 │  │                 │              │ 
│  │ meetingbank     │  │                 │  │                 │              │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘              │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                         LLM 大模型                               │        │
│  │              Ollama (Qwen2.5) / OpenAI (GPT-4)                  │        │
│  └─────────────────────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 目录结构

```
Prompt-composer/
│
├── app.py                          # Gradio Web 应用入口
├── config.py                        # 全局配置（路径、参数、开关）
│
├── core/                            # 核心算法模块
│   ├── __init__.py
│   │
│   ├── compression/                 # 📌 压缩模块
│   │   ├── __init__.py
│   │   ├── compressor.py            # HardCompressor (MOOSComp/LLMLingua2)
│   │   ├── merger.py                 # PromptMerger (语义合并)
│   │   ├── preprocessors.py          # DocumentPreprocessor (6种预处理策略)
│   │   ├── compression_strategy.py   # CompressionStrategy (自适应选择)
│   │   ├── keyword_retriever.py      # KeywordRetriever (BM25关键词检索)
│   │   ├── word_priority_compressor.py # WordPriorityCompressor (词级压缩)
│   │   ├── word_priority_model.py    # WordPriorityModel (LSTM优先级模型)
│   │   ├── hybrid_compressor.py      # HybridCompressor (混合压缩，待优化)
│   │   ├── attention_merger.py        # AttentionBasedMerger (软压缩)
│   │   ├── token_merger.py           # InferenceTokenMerger
│   │   ├── learnable_merger.py       # LearnableTokenMerger
│   │   ├── cross_segment_aggregator.py
│   │   └── hierarchical_aggregator.py
│   │
│   ├── retrieval/                    # 📌 检索模块
│   │   ├── __init__.py
│   │   ├── indexer.py                # Indexer (FAISS向量索引)
│   │   ├── query_expander.py         # QueryExpander (查询扩展)
│   │   ├── result_compressor.py      # ResultCompressor (结果去重)
│   │   ├── context_builder.py        # ContextBuilder (上下文构建)
│   │   ├── dynamic_allocator.py      # DynamicTokenAllocator (动态分配)
│   │   └── t5_fusion.py              # T5FusionEngine (T5语义融合)
│   │
│   ├── engine/                       # 📌 引擎模块
│   │   ├── __init__.py
│   │   ├── qa_engine.py              # QAEngine (主引擎)
│   │   └── summarizer.py             # DialogSummarizer (对话摘要)
│   │
│   ├── history/                      # 📌 历史管理模块
│   │   ├── __init__.py
│   │   ├── history_manager.py        # HistoryManager
│   │   ├── tree_history.py            # MultiSourceTree (多源聚合树)
│   │   └── textrank_summarizer.py    # TextRankSummarizer
│   │
│   ├── storage/                      # 📌 存储模块
│   │   ├── __init__.py
│   │   ├── vector_store.py           # VectorStore (基础向量库)
│   │   └── enhanced_vector_store.py  # EnhancedVectorStore (增强版)
│   │
│   ├── io/                           # 📌 IO模块
│   │   ├── __init__.py
│   │   └── document_reader.py        # DocumentReader (多格式读取)
│   │
│   ├── utils/                        # 📌 工具模块
│   │   ├── __init__.py
│   │   └── utils.py                  # split_sentences, count_tokens 等
│   │
│   ├── visualization/                 # 📌 可视化模块
│   │   ├── __init__.py
│   │   └── tree_visualizer.py        # visualize_tree
│   │
│   └── model/                         # 📌 模型客户端
│       ├── __init__.py
│       └── llm_client.py              # LLMClient (Ollama/OpenAI)
│
├── model/                            # 预训练模型权重
│   ├── compression_bert_mooscomp_news/    # MOOSComp 硬压缩
│   ├── llmlingua-2-bert-base-multilingual-cased-meetingbank/ # LLMLingua2
│   ├── word_priority_model/                # 词级优先级模型
│   ├── word_priority_model_mixed/          # 混合微调词级模型
│   ├── paraphrase-multilingual-MiniLM-L12-v2/ # 句子编码器
│   ├── mengzi-t5-base/                      # T5 基座
│   ├── mengzi-t5-finetuned-fusion/         # T5 LoRA 融合
│   ├── sentence_labeling_adapter_cosine/   # 摘要 Adapter
│   └── attention_merger_adapter/           # 软压缩 Adapter
│
├── eval_results/                     # 评测结果输出目录
├── SYS/                              # 系统文档目录
│   └── ARCHITECTURE.md              # 本文档
│
└── Model_traing/                     # 训练脚本与数据
    ├── data/longpaper/short_4k.json # 测试数据
    └── word_priority/                # 词级优先级训练
```

---

## 🔧 核心模块详解

### 1. 压缩模块 (Compression)

#### 1.1 HardCompressor - BERT硬压缩器

**文件**: `core/compression/compressor.py`

**功能**: 基于BERT Token分类的硬压缩，按比例保留重要Token

**模型支持**:
| model_type | 模型路径 | 说明 |
|------------|----------|------|
| `mooscomp` | `compression_bert_mooscomp_news` | 新闻领域MOOSComp |
| `llmlingua2` | `llmlingua-2-bert-base-multilingual-cased-meetingbank` | 会议bank |

**核心方法**:
```python
class HardCompressor:
    def compress(self, sentence, compression_ratio=0.7):
        """
        单句压缩：返回 (压缩文本, 原始token数, 压缩后token数)
        """
    def compress_batch(self, sentences, compression_ratio=0.7):
        """批量压缩"""
    def compress_to_target_tokens(self, text, target_tokens):
        """压缩到目标token数（分块策略）"""
```

**压缩原理**:
1. Tokenize输入文本
2. BERT前向传播获取每个Token的保留概率
3. 按概率降序选择Top-K Token
4. 解码保留的Token得到压缩文本

---

#### 1.2 PromptMerger - 语义合并器

**文件**: `core/compression/merger.py`

**功能**: 使用SentenceTransformer编码+余弦相似度合并相似片段

**核心方法**:
```python
class PromptMerger:
    def process(self, fragments, do_merge=True):
        """
        输入: fragment列表
        输出: {'fragments': [...], 'vectors': np.array}
        合并相似片段（阈值0.8）并融合向量
        """
```

---

#### 1.3 DocumentPreprocessor - 文档预处理器

**文件**: `core/compression/preprocessors.py`

**功能**: 封装6种文档预处理策略

| 策略 | 方法名 | 适用场景 | 说明 |
|------|--------|---------|------|
| 独立分段压缩 | `preprocess_independent` | 标准RAG | 分句→压缩→合并→向量化 |
| 全局压缩+片段检索 | `preprocess_global_compress` | 长文档极端压缩 | 先压缩全文，再按目标Token切分 |
| Token合并 | `preprocess_token_merge` | 减少检索向量数 | 语义合并后再压缩 |
| 可学习合并 | `preprocess_learnable_merge` | 端到端优化 | 使用可学习合并器 |
| 层次化聚合 | `preprocess_hierarchical` | 跨段关联建模 | 两层Transformer |
| 增强型压缩 | `preprocess_enhanced` | 问题感知压缩 | 问题相关段落优先 |

**核心方法**:
```python
class DocumentPreprocessor:
    def __init__(self, model_path, encoder_name, device,
                 compression_ratio=0.7, batch_size=32):
        self.compressor = HardCompressor(...)
        self.merger = PromptMerger(...)

    def preprocess_independent(self, docs_dict, compression_ratio):
        """标准独立分段压缩"""
        # 1. 分句
        # 2. 批量压缩
        # 3. 语义合并
        # 4. 向量化
        return fragments, vectors, doc_ids, doc_orders
```

---

#### 1.4 CompressionStrategy - 自适应策略选择

**文件**: `core/compression/compression_strategy.py`

**功能**: 根据文档特征自动选择最优预处理策略

**策略选择逻辑**:
| 条件 | 策略 |
|------|------|
| 文档≤40K tokens | `full_compress` 全文压缩直答 |
| 文档>40K tokens 且 极端压缩 | `global_compress` 全局压缩+检索 |
| 文档>40K tokens | `independent` 标准分段压缩 |

---

#### 1.5 KeywordRetriever - 关键词检索器

**文件**: `core/compression/keyword_retriever.py`

**功能**: BM25倒排索引检索（压缩比≤0.7时启用）

```python
class KeywordRetriever:
    def build_index(self, fragments):
        """构建jieba分词+BM25索引"""
    def retrieve(self, query, top_k=5):
        """检索相关片段"""
```

---

#### 1.6 WordPriorityCompressor - 词级优先级压缩

**文件**: `core/compression/word_priority_compressor.py`

**功能**: 基于词级优先级模型的压缩，保留关键词汇

**模型结构**: BERT + BiLSTM + Sigmoid回归头

```python
class WordPriorityCompressor:
    def compress(self, text, compression_ratio=0.7):
        """返回压缩后的字符串"""
    def compress_batch(self, texts, compression_ratio):
        """批量压缩"""
```

---

### 2. 检索模块 (Retrieval)

#### 2.1 Indexer - FAISS向量索引

**文件**: `core/retrieval/indexer.py`

**功能**: 稠密向量索引和检索

```python
class Indexer:
    def build(self, vectors, fragments):
        """构建FAISS IndexFlatIP索引"""
    def search(self, query_vector, top_k):
        """返回 (scores, indices)"""
```

---

#### 2.2 QueryExpander - 查询扩展器

**文件**: `core/retrieval/query_expander.py`

**功能**: 基于同义词和实体识别的查询扩展（无LLM依赖）

```python
class QueryExpander:
    def expand(self, query, max_expansions=3):
        """返回扩展后的查询列表"""
    def extract_key_entities(self, query):
        """提取关键实体（人名/地名/机构名）"""
```

---

#### 2.3 T5FusionEngine - T5语义融合引擎

**文件**: `core/retrieval/t5_fusion.py`

**功能**: 使用LoRA微调的T5模型对高相似度片段进行语义融合

```python
class T5FusionEngine:
    def fuse(self, chunks: List[Dict]) -> List[Dict]:
        """
        输入: [{'content': '...', 'score': 0.9}, ...]
        输出: 融合后的片段列表
        融合阈值: 0.75
        """
```

---

#### 2.4 ContextBuilder - 上下文构建器

**文件**: `core/retrieval/context_builder.py`

**功能**: 信息密度加权 + Token预算分配 + 句子级截断

```python
class ContextBuilder:
    def build_context(self, retrieved_chunks, token_budget):
        """
        1. 计算每个片段的信息密度
        2. 按密度和相关性加权
        3. 贪心选取填充Token预算
        4. 句子级截断保证完整性
        """
```

---

### 3. 引擎模块 (Engine)

#### 3.1 QAEngine - 问答引擎

**文件**: `core/engine/qa_engine.py`

**功能**: 整合检索、压缩、生成、历史管理的核心引擎

**检索模式**:

| 压缩比 | 检索模式 | 说明 |
|--------|---------|------|
| ≤0.7 | 关键词锚定检索 | BM25 + KeywordRetriever |
| >0.7 | 混合检索 | BM25 + 稠密向量 + T5融合 |

**核心方法**:
```python
class QAEngine:
    def answer(self, question, documents, alpha, use_expansion,
                use_compression, use_smart_context):
        """
        混合检索流程:
        1. 查询扩展 (QueryExpander)
        2. BM25初筛 + 向量检索
        3. T5融合去重
        4. 动态Token分配
        5. 上下文构建
        6. LLM生成
        """

    def answer_direct(self, question, context):
        """压缩直答模式"""

    def add_turn(self, question, answer, retrieved_chunks):
        """添加对话历史"""
```

---

#### 3.2 DialogSummarizer - 对话摘要器

**文件**: `core/engine/summarizer.py`

**功能**: 对话历史摘要，支持Adapter和TextRank降级

---

### 4. 历史管理模块 (History)

#### 4.1 HistoryManager - 历史管理器

**文件**: `core/history/history_manager.py`

**功能**: 多源聚合树 + TextRank摘要 + 增强向量库

```python
class HistoryManager:
    def add_turn(self, question, answer, retrieved_chunks):
        """添加问答轮次，自动合并相似信息节点"""

    def get_context_for_llm(self):
        """获取用于LLM的历史上下文"""

    def get_fallback_text(self):
        """降级文本（用于UI展示）"""
```

---

### 5. 存储模块 (Storage)

#### 5.1 EnhancedVectorStore - 增强向量库

**文件**: `core/storage/enhanced_vector_store.py`

**功能**: 支持自动融合的向量数据库

```python
class EnhancedVectorStore:
    def add(self, content, source_id, metadata, auto_fusion=True):
        """添加记录，自动融合高相似度内容"""
    def search(self, query, top_k=3):
        """检索并返回融合后的结果"""
```

---

## 📊 数据流图

### 问答流程

```
用户问题
    │
    ▼
┌───────────────────────────────────────────────────────────────┐
│                     QAEngine.answer()                          │
│                                                               │
│  ┌─────────────┐                                              │
│  │QueryExpander│ 查询扩展（同义词、实体识别）                    │
│  └──────┬──────┘                                              │
│         │                                                     │
│         ▼                                                     │
│  ┌─────────────┐     ┌─────────────┐                          │
│  │   BM25      │────▶│   FAISS     │  混合检索                │
│  │   关键词检索 │     │   向量检索   │                          │
│  └──────┬──────┘     └──────┬──────┘                          │
│         │                   │                                 │
│         └─────────┬─────────┘                                 │
│                   ▼                                           │
│         ┌─────────────────┐                                    │
│         │  T5FusionEngine │ 语义融合去重                        │
│         └────────┬────────┘                                    │
│                  ▼                                             │
│         ┌─────────────────┐                                    │
│         │DynamicAllocator │ 动态Token分配                       │
│         └────────┬────────┘                                    │
│                  ▼                                             │
│         ┌─────────────────┐                                    │
│         │ ContextBuilder │ 上下文构建                          │
│         └────────┬────────┘                                    │
│                  ▼                                             │
│         ┌─────────────────┐                                    │
│         │   LLMClient     │ LLM生成答案                         │
│         └────────┬────────┘                                    │
│                  ▼                                             │
│         ┌─────────────────┐                                    │
│         │HistoryManager  │ 更新对话历史                         │
│         └─────────────────┘                                    │
└───────────────────────────────────────────────────────────────┘
    │
    ▼
答案 + Token统计
```

### 文档预处理流程

```
上传文档
    │
    ▼
┌───────────────────────────────────────────────────────────────┐
│               DocumentPreprocessor.preprocess_*()             │
│                                                               │
│  ┌─────────────┐                                              │
│  │split_sentences│ 分句（。！？）                               │
│  └──────┬──────┘                                              │
│         ▼                                                     │
│  ┌─────────────┐                                              │
│  │HardCompressor│ BERT硬压缩（按比例保留Token）                 │
│  └──────┬──────┘                                              │
│         ▼                                                     │
│  ┌─────────────┐                                              │
│  │ PromptMerger │ 语义合并（相似片段融合）                       │
│  └──────┬──────┘                                              │
│         ▼                                                     │
│  ┌─────────────┐                                              │
│  │   Encoder   │ SentenceTransformer向量化                     │
│  └──────┬──────┘                                              │
│         ▼                                                     │
│  fragments[], vectors[], doc_ids[], doc_orders[]              │
└───────────────────────────────────────────────────────────────┘
```

---

## 🔌 配置参数 (config.py)

### 压缩参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `HARD_COMPRESSION_RATIO` | 0.7 | 硬压缩保留比例 |
| `MODEL_PATH` | `compression_bert_mooscomp_news` | MOOSComp模型路径 |
| `ENCODER_NAME` | `paraphrase-multilingual-MiniLM-L12-v2` | 编码器路径 |

### 检索参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `HYBRID_ALPHA` | -0.8 | 混合检索BM25权重 |
| `BM25_TOP_K` | 100 | BM25初筛候选数 |
| `TOP_K` | 10 | 最终返回数 |
| `RELEVANCE_THRESHOLD` | 0.4 | 相关性阈值 |

### 历史参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MAX_CONTEXT_TOKENS` | 500 | 历史上下文最大Token |
| `CONTEXT_WINDOW_SIZE` | 2000 | 模型窗口大小 |
| `HISTORY_COMPRESS_THRESHOLD` | 5 | 触发摘要的轮数 |

### T5融合参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `CHUNK_FUSER_MODEL` | `mengzi-t5-finetuned-fusion` | 融合模型路径 |
| `CHUNK_FUSION_THRESHOLD` | 0.75 | 融合相似度阈值 |

---

## 🚀 使用示例

### 1. 基础问答 (app.py)

```python
from app import process_documents_advanced, answer_wrapper_advanced

# 1. 上传并预处理文档
fragments, vectors, doc_ids, doc_orders, stats, _ = process_documents_advanced(
    files=[...],
    text_input="文档内容...",
    compression_ratio=0.7,
    preprocessing_mode="auto"
)

# 2. 问答
answer, token_stats, history_text = answer_wrapper_advanced(
    question="问题",
    fragments=fragments,
    vectors=vectors,
    doc_ids=doc_ids,
    doc_orders=doc_orders,
    config_state={"compression_ratio": 0.7, ...}
)
```

### 2. 独立使用压缩器

```python
from core.compression import HardCompressor

compressor = HardCompressor(model_type="mooscomp", device="cpu")
compressed = compressor.compress("原始文本", compression_ratio=0.5)
```

### 3. 批量文档预处理

```python
from core.compression import DocumentPreprocessor

preprocessor = DocumentPreprocessor(
    model_path="model/compression_bert_mooscomp_news",
    encoder_name="model/paraphrase-multilingual-MiniLM-L12-v2",
    device="cpu",
    compression_ratio=0.7
)

docs_dict = {"文档1": "内容...", "文档2": "内容..."}
fragments, vectors, doc_ids, doc_orders = preprocessor.preprocess_independent(docs_dict)
```

---

## 📝 维护说明

**核心原则**:
1. **模块独立性**: 各模块可独立使用，通过接口交互
2. **配置外置**: 所有参数集中在 `config.py`
3. **模型本地化**: 强制 `local_files_only=True` 避免联网
4. **渐进式压缩**: 关键词检索 → 混合检索 → 全文直答

**扩展指南**:
- 新增压缩策略: 在 `preprocessors.py` 添加 `preprocess_*` 方法
- 新增检索模式: 在 `qa_engine.py` 添加 `answer_*` 方法
- 新增模型: 在 `config.py` 添加路径配置

---

**文档版本**: v3.0
**最后更新**: 2026-05-03
