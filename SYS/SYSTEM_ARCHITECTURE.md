# Prompt Composer 系统架构文档

## 📋 目录

- [系统概述](#系统概述)
- [系统架构](#系统架构)
- [核心模块](#核心模块)
- [模型与组件](#模型与组件)
- [工作流程](#工作流程)
- [训练流程](#训练流程)
- [配置说明](#配置说明)

---

## 系统概述

Prompt Composer 是一个基于 RAG（检索增强生成）的智能问答系统，集成了**文档压缩**、**混合检索**、**对话历史管理**和**文本融合**四大核心能力。

### 核心特性

- ✅ **智能文档压缩**：BERT 硬压缩 + Attention 软压缩
- ✅ **混合检索**：BM25 + 稠密向量融合检索
- ✅ **多源信息融合**：T5 语义融合去重
- ✅ **对话历史摘要**：基于 Adapter 的历史压缩
- ✅ **动态 Token 分配**：智能上下文优化
- ✅ **多 LLM 后端**：支持 Ollama / OpenAI 兼容 API

---

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    用户交互层 (Gradio)                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │
│  │ 文档上传  │  │ 问题输入  │  │ 检索调节  │  │ 答案展示 │ │
│  └──────────┘  └──────────┘  └──────────┘  └─────────┘ │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                  业务逻辑层 (QAEngine)                    │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐ │
│  │ 文档处理流程  │  │ 检索与融合   │  │  LLM 调用     │ │
│  └──────────────┘  └──────────────┘  └───────────────┘ │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                    核心算法层 (Core)                      │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌──────────────┐  │
│  │ 压缩模块 │ │ 检索模块 │ │ 存储模块 │ │ 历史管理模块  │  │
│  └─────────┘ └─────────┘ └─────────┘ └──────────────┘  │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                    模型层 (Models)                        │
│  ┌──────┐ ┌──────┐ ┌─────┐ ┌────┐ ┌─────┐ ┌────────┐  │
│  │ BERT │ │ T5  │ │ST  │ │FAISS│ │LLM  │ │Adapter │  │
│  └──────┘ └──────┘ └─────┘ └────┘ └─────┘ └────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 核心模块

### 1. 文档压缩模块 (`core/compression/`)

#### 1.1 HardCompressor（硬压缩）

**功能**：基于 BERT 的 Token 级压缩

**工作原理**：
```
原文 → BERT 编码 → Token 分类器 → 保留重要 Token → 压缩文本
```

**特点**：
- 直接删除不重要 Token
- 压缩比：50-70%
- 输出：可读文本
- 训练方式：句子重要性标注 + 微调

**文件**：
- `compressor.py`：硬压缩实现
- `compression_bert_model/`：预训练模型

#### 1.2 SoftCompressorWithAdapter（软压缩 - 新）

**功能**：基于注意力机制的向量级压缩

**工作原理**：
```
原文 → BERT 编码 → Attention Merger → [64, 768] 向量表示
```

**特点**：
- 输出语义向量（非文本）
- 压缩比精确可控（通过 prototype 数量）
- 保留完整语义信息
- 适用于下游任务（问答、检索）

**组件**：
- `attention_merger.py`：Attention Merger 实现
- `attention_merger_adapter/`：训练好的 Adapter 权重

**训练流程**：
```bash
# 1. 训练 Adapter
python train_attention_merger.py \
  --bert_model ../compression_bert_model \
  --train_data train.json \
  --val_data val.json \
  --output_dir ../attention_merger_adapter \
  --num_prototypes 64 \
  --num_epochs 10
```

#### 1.3 T5Decoder（T5 解码器 - 待训练）

**功能**：将软压缩向量解码为可读文本

**工作原理**：
```
原文 → BERT → Adapter → [64, 768] → 投影层 → T5 → 压缩文本
```

**训练数据**：需要 `(原文, 压缩文本)` 对

**脚本**：
- `train_t5_decoder.py`：训练脚本
- `test_t5_decoder.py`：测试脚本
- `annotate_fusion_dataset.py`：数据标注脚本

---

### 2. 检索模块 (`core/retrieval/`)

#### 2.1 Indexer（索引器）

**功能**：构建 FAISS 向量索引

**工作流程**：
```
文档 → 分句 → 压缩 → 编码 → FAISS 索引
```

**支持**：
- 稠密向量检索（Sentence Transformer）
- BM25 关键词检索
- 混合检索（加权融合）

#### 2.2 DynamicAllocator（动态分配器）

**功能**：智能 Token 分配

**策略**：
- 根据问题相关性分配 Token 预算
- 优先保留高相关度文档
- 动态调整上下文窗口

#### 2.3 ContextBuilder（上下文构建器）

**功能**：构建 LLM 输入上下文

**流程**：
```
问题 → 检索片段 → 相关性过滤 → 同文档扩展 → 上下文
```

#### 2.4 T5FusionEngine（T5 融合引擎）

**功能**：检索结果语义融合去重

**工作原理**：
```
多个检索片段 → 相似度计算 → 分组 → T5 融合 → 去重后片段
```

**特点**：
- 识别语义重叠
- 删除重复信息
- 保留所有关键事实
- 生成流畅文本

**模型**：`mengzi-t5-base-finetuned-fusion-merged`

**配置**：
```python
CHUNK_FUSER_MODEL = "model/mengzi-t5-base-finetuned-fusion-merged"
USE_CHUNK_FUSION = True
CHUNK_FUSION_THRESHOLD = 0.75  # 融合阈值
```

---

### 3. 历史管理模块 (`core/history/`)

#### 3.1 HistoryManager（历史管理器）

**功能**：管理对话历史

**策略**：
- 滑动窗口：保留最近 N 轮
- 自动压缩：超过阈值触发摘要
- 相关性过滤：只保留相关历史

#### 3.2 Summarizer（摘要器）

**功能**：压缩对话历史

**架构**：
```
对话历史 → BERT 编码 → Adapter → 关键句选择 → 摘要
```

**模型**：
- Base: `compression_bert_model`
- Adapter: `sentence_labeling_adapter_cosine`

**训练方式**：对比学习 + 重建损失

---

### 4. 存储模块 (`core/storage/`)

#### 4.1 VectorStore（向量存储）

**功能**：管理 FAISS 索引和元数据

**特性**：
- 支持多文档索引
- 文档级元数据管理
- 增量更新

#### 4.2 EnhancedVectorStore（增强向量存储）

**功能**：扩展版向量存储

**新增**：
- 压缩向量缓存
- 快速检索优化
- 批量操作支持

---

### 5. LLM 客户端 (`core/model/`)

#### 5.1 LLMClient

**支持后端**：
- Ollama（本地部署）
- OpenAI 兼容 API（DashScope、Azure 等）

**配置示例**：
```python
# Ollama
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:3b"

# DashScope
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
model = "qwen-max"
api_key = "sk-xxx"
```

---

## 模型与组件

### 模型清单

| 模型名称 | 用途 | 位置 | 状态 |
|---------|------|------|------|
| **compression_bert_model** | 文档硬压缩 | `model/compression_bert_model/` | ✅ 已训练 |
| **paraphrase-multilingual-MiniLM-L12-v2** | 句子编码（检索） | `model/paraphrase-multilingual-MiniLM-L12-v2/` | ✅ 预训练 |
| **sentence_labeling_adapter_cosine** | 对话历史摘要 | `model/sentence_labeling_adapter_cosine/` | ✅ 已训练 |
| **mengzi-t5-finetuned-fusion** | 检索结果融合 (LoRA) | `model/mengzi-t5-finetuned-fusion/` | ✅ 已训练 (LoRA Adapter) |
| **mengzi-t5-base** | T5 融合基础模型 | `model/mengzi-t5-base/` | ✅ 预训练 |
| **attention_merger_adapter** | 软压缩 Adapter | `model/attention_merger_adapter/` | ✅ 已训练 |
| **t5_decoder_adapter** | T5 解码器 | 待训练 | ⏳ 待标注数据 |

### 模型详细说明

#### 1. BERT 压缩模型

- **架构**：BERT-base + Token 分类头
- **参数量**：~110M
- **训练数据**：新闻句子 + 重要性标注
- **功能**：Token 级重要性评分
- **输出**：压缩后的文本（删除低分 Token）

#### 2. Sentence Transformer

- **架构**：Multi-Lingual MiniLM
- **参数量**：~120M
- **维度**：384 维
- **功能**：句子语义编码
- **用途**：稠密检索、相似度计算

#### 3. 对话摘要 Adapter

- **架构**：BERT + LoRA Adapter
- **基础模型**：compression_bert_model
- **参数量**：~2.8M（可训练）
- **训练方式**：对比学习
- **功能**：选择关键句子

#### 4. T5 融合模型 (LoRA)

- **架构**：Mengzi-T5-Base + LoRA Adapter
- **基础模型参数量**：~250M
- **Adapter 参数量**：~3.5M (可训练)
- **训练数据**：高相似度文本对 + 融合标注 (1325条训练集)
- **功能**：语义去重 + 信息融合
- **生成策略**：Beam Search (num_beams=4, length_penalty=0.8)
- **模型路径**：`model/mengzi-t5-finetuned-fusion/`
- **加载方式**：PeftModel.from_pretrained(base_model, adapter_path)
- **评测表现**：压缩比 ~50%，关键信息保留率 >95%

#### 5. Attention Merger Adapter（新）

- **架构**：多头注意力 + 原型向量
- **参数量**：~2.8M
- **原型数量**：64
- **训练方式**：对比学习 + 重建损失 + 多样性损失
- **功能**：将变长 Token 序列压缩为固定数量向量
- **输出**：[64, 768] 张量

#### 6. T5 解码器（待训练）

- **架构**：Mengzi-T5-Base + 投影层
- **输入**：[64, 768] 压缩向量
- **输出**：压缩文本
- **训练数据**：需 API 标注（约 5000 条）
- **预计成本**：¥250

---

## 工作流程

### 文档处理流程

```
1. 上传文档（.txt / 直接输入）
   ↓
2. 分句（按 。！？;； 分割）
   ↓
3. BERT 硬压缩（删除冗余 Token）
   ↓
4. Sentence Transformer 编码（生成向量）
   ↓
5. 语义合并（相似度 > 0.9 的合并）
   ↓
6. 构建 FAISS 索引
   ↓
7. 生成文档摘要
```

### 问答流程

```
1. 用户输入问题
   ↓
2. 话题相关性判断
   ├─ 与文档摘要比较 → 低于阈值跳过检索
   └─ 与历史摘要比较 → 低于阈值不加入上下文
   ↓
3. 混合检索
   ├─ BM25 关键词检索（Top-50）
   ├─ 稠密向量检索（Top-50）
   └─ 加权融合（alpha=0.5）
   ↓
4. T5 融合去重
   ├─ 计算片段相似度
   ├─ 相似度 > 0.75 的分组
   └─ T5 融合每组 → 去重后片段
   ↓
5. 动态 Token 分配
   ├─ 按相关度排序
   ├─ 分配 Token 预算
   └─ 同文档上下文扩展
   ↓
6. 历史语境构建
   ├─ 检索相关历史摘要
   └─ 限制 Token 数（< 150）
   ↓
7. 构建 LLM Prompt
   ├─ 系统指令
   ├─ 检索片段
   ├─ 历史语境
   └─ 用户问题
   ↓
8. 调用 LLM 生成答案
   ↓
9. 生成对话摘要（压缩历史）
   ↓
10. 返回答案 + Token 统计
```

---

## 训练流程

### 已完成的训练

#### 1. BERT 硬压缩模型

```bash
# 数据标注
python generate_data_with_api.py \
  --api_key "sk-xxx" \
  --max_samples 5000

# 训练
python train_compression_bert.py \
  --model_path bert-base-chinese \
  --train_data train.json \
  --output_dir model/compression_bert_model
```

#### 2. 对话摘要 Adapter

```bash
# 训练
python train_summarizer_adapter.py \
  --bert_model model/compression_bert_model \
  --train_data dialogue_train.json \
  --output_dir model/sentence_labeling_adapter_cosine
```

#### 3. T5 融合模型

```bash
# 数据标注
python annotate_fusion_dataset.py \
  --api_key "sk-xxx" \
  --max_samples 5000

# 训练
python train_t5_fusion.py \
  --model_path mengzi-t5-base \
  --train_data t5_fusion_train.json \
  --output_dir model/mengzi-t5-base-finetuned-fusion-merged
```

#### 4. Attention Merger Adapter

```bash
# 服务器训练
cd ~/t5_training/bert/attention_merger_pkg

python train_attention_merger.py \
  --bert_model ../compression_bert_model \
  --train_data train.json \
  --val_data val.json \
  --output_dir ../attention_merger_adapter \
  --num_prototypes 64 \
  --num_epochs 10
```

### 待完成的训练

#### T5 解码器

```bash
# 1. 数据标注（待执行）
python annotate_fusion_dataset.py \
  --api_key "sk-b4df3774c84349ae8ee77cb86ac021df" \
  --max_samples 5000 \
  --output_dir data/training_data

# 2. 训练（待执行）
python train_t5_decoder.py \
  --bert_model ../compression_bert_model \
  --adapter_path ../attention_merger_adapter/best_adapter.pth \
  --t5_model /root/t5_training/mengzi-t5-base \
  --train_data t5_decoder_train.json \
  --val_data t5_decoder_val.json \
  --output_dir ../t5_decoder_adapter \
  --num_epochs 10
```

---

## 配置说明

### 全局配置 (`config.py`)

```python
# 模型路径
MODEL_PATH = "../model/compression_bert_model"
ENCODER_NAME = "model/paraphrase-multilingual-MiniLM-L12-v2"
SUMMARIZER_BASE = "model/compression_bert_model"
SUMMARIZER_ADAPTER = "model/sentence_labeling_adapter_cosine/final_adapter"
CHUNK_FUSER_MODEL = "model/mengzi-t5-finetuned-fusion"  # LoRA Adapter

# 检索参数
TOP_K = 5  # 最终检索片段数
HYBRID_ALPHA = 0.5  # 稠密检索权重
BM25_TOP_K = 50  # 初筛候选数
RELEVANCE_THRESHOLD = 0.5  # 话题相关性阈值
CHUNK_FUSION_THRESHOLD = 0.75  # 融合分组阈值

# 历史压缩参数
HISTORY_COMPRESS_THRESHOLD = 5  # 触发压缩的轮数
MAX_SUMMARY_TOKENS = 500  # 摘要最大 Token 数
MAX_HISTORY_LEN = 500  # 历史窗口长度

# 上下文参数
SELECTED_SUMMARIES_COUNT = 3  # 相关摘要数量
MAX_CONTEXT_TOKENS = 150  # 历史语境 Token 上限
CONTEXT_WINDOW_SIZE = 4096  # 模型上下文窗口
```

### 服务器路径

```
/root/t5_training/bert/
├── compression_bert_model/              # BERT 基础模型
├── attention_merger_pkg/                # 训练代码
│   ├── train_attention_merger.py
│   ├── test_attention_merger.py
│   ├── train.json
│   └── val.json
└── attention_merger_adapter/            # 训练好的 Adapter
    ├── best_adapter.pth
    ├── final_adapter.pth
    └── training_history.json
```

---

## 性能指标

### 压缩性能

| 方法 | 压缩比 | 输出类型 | 关键信息保留 |
|------|--------|---------|-------------|
| HardCompressor | 50-70% | 文本 | 中等 |
| SoftCompressor | 固定 64 token | 向量 | 高 |
| SoftCompressor + T5 | 60-75% | 文本 | 高（待验证） |

### 检索性能

| 模式 | 准确率 | 召回率 | 适用场景 |
|------|--------|--------|---------|
| BM25 | 高 | 中 | 精确匹配 |
| 稠密向量 | 中 | 高 | 语义匹配 |
| 混合（α=0.5） | 高 | 高 | 通用场景 |

### 融合性能

| 指标 | 值 | 说明 |
|------|-----|------|
| 融合准确率 | ~85% | 去重准确率 |
| 信息保留率 | ~95% | 关键信息不丢失 |
| 幻觉率 | < 5% | 无中生有比例 |
| 平均压缩比 | ~50% | 融合后文本长度 |

### 系统评测基线

#### Qwen2.5-3B 本地评测结果
- **测试集**: news_testset.json (10篇文档, 50个QA对)
- **生成模型**: qwen2.5:3b (Ollama)
- **T5融合**: 已启用 (mengzi-t5-finetuned-fusion LoRA)

| 维度 | 得分 | 说明 |
|------|------|------|
| 准确性 | 71.5/100 | 良好 |
| 完整性 | 72.3/100 | 良好 |
| 相关性 | 86.0/100 | 优秀 |
| 简洁性 | 79.2/100 | 良好 |
| **综合得分** | **77.25/100** | **整体表现良好** |

- **检索统计**: 平均 3.0 chunks/问题

#### Qwen2.5-7B 服务器评测
- **模型路径**: `/root/models/Qwen2.5-7B-Instruct`
- **状态**: 评估脚本已创建 (`eval_qwen2.5-7b.py`)
- **预期提升**: 综合得分 81-84/100 (+4~7分)

---

## 依赖环境

### Python 依赖

```bash
pip install -r requirements.txt
pip install -r requirements_optional.txt  # 可选依赖
```

### 核心依赖

- `transformers>=4.30.0`：模型加载
- `faiss-cpu` / `faiss-gpu`：向量检索
- `sentence-transformers`：句子编码
- `adapter-transformers`：Adapter 支持
- `openai`：API 调用
- `gradio`：Web 界面
- `protobuf==3.20.3`：T5 tokenizer 依赖

### 硬件要求

- **CPU**：4 核+
- **内存**：8GB+（推荐 16GB）
- **GPU**（可选）：4GB+ VRAM（加速推理）

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备模型

```bash
# 确保以下模型存在
ls model/compression_bert_model/
ls model/paraphrase-multilingual-MiniLM-L12-v2/
ls model/sentence_labeling_adapter_cosine/
ls model/mengzi-t5-finetuned-fusion/  # LoRA Adapter
ls model/mengzi-t5-base/              # T5 基础模型
```

### 3. 启动应用

```bash
python app.py
```

访问 `http://localhost:7860`

---

## 常见问题

### Q1: T5 融合不生效？

检查配置：
```python
USE_CHUNK_FUSION = True  # 必须为 True
CHUNK_FUSION_THRESHOLD = 0.75  # 阈值不能太高
```

### Q2: 显存不足？

减少批次大小或关闭 GPU：
```python
device = "cpu"  # 强制使用 CPU
```

### Q3: 如何切换 LLM 后端？

在界面"模型配置"面板中选择：
- Ollama：填写地址和模型名
- OpenAI 兼容：填写 API URL 和 Key

---

## 更新日志

### 2026-04-12 (最新)
- ✅ 完成 T5 融合模型 LoRA 训练 (mengzi-t5-finetuned-fusion)
- ✅ 更新 T5 融合引擎支持 LoRA Adapter 加载
- ✅ 创建 Qwen2.5-7B 评估脚本 (eval_qwen2.5-7b.py)
- ✅ 完成 Qwen2.5-3B 本地评测 (综合得分 77.25/100)
- ✅ 创建新闻领域测试集 (news_testset.json, 50个QA对)

### 2024-04-11
- ✅ 完成 Attention Merger Adapter 训练
- ✅ 创建 T5 解码器训练脚本
- ✅ 创建融合数据标注脚本
- ⏳ 待标注 T5 解码器训练数据

### 2024-04-10
- ✅ 完成 T5 融合模型训练
- ✅ 集成到检索流程
- ✅ 添加融合质量验证

### 2024-04-09
- ✅ 完成对话摘要 Adapter 训练
- ✅ 实现动态 Token 分配
- ✅ 优化上下文构建逻辑

---

## 联系方式

如有问题或建议，请提交 Issue。
