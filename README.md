```markdown
# Prompt Composer 工具

基于 BERT 硬压缩 + 混合检索 + 对话摘要的智能问答系统，支持多文档处理、语义/关键词检索切换，并可对接 Ollama 或 OpenAI 兼容的 LLM 服务（如阿里云 DashScope）。

## 功能特性

多文档压缩与索引：上传多个文本文件，自动分句、BERT 硬压缩、语义合并，构建 FAISS 向量索引。
混合检索：结合 BM25（关键词）和稠密向量（语义）进行加权融合，支持滑块实时调节检索倾向（-1 完全关键词 → 0 平衡 → 1 完全语义）。
对话历史摘要：每轮问答后自动生成摘要，后续对话仅基于摘要 + 检索片段，极大节省 LLM Token 消耗。
多后端支持：通过界面配置 Ollama 或 OpenAI 兼容 API（如阿里云 DashScope），无需修改代码即可切换模型。
话题相关性判断：自动检测问题与文档/摘要的语义相似度，无关内容不会污染上下文。
同文档上下文扩展：检索到的片段自动附带同一文档内的前后句，保证回答连贯性。
实时 Token 统计：显示输入/输出 Token 数及检索片段 Token 数，便于监控成本。

## 项目结构
Prompt-composer/
│
├── app.py # Gradio 主程序（含界面与业务逻辑）
├── config.py # 全局配置（模型路径、检索参数、阈值等）
├── requirements.txt # Python 依赖
├── README.md # 项目说明
│
├── core/ # 核心算法模块
│ ├── init.py
│ ├── compressor.py # BERT 硬压缩
│ ├── merger.py # 语义合并
│ ├── indexer.py # FAISS 索引与检索
│ ├── llm_client.py # LLM 客户端（支持 Ollama / OpenAI 兼容）
│ ├── summarizer.py # 对话摘要模型（基于 BERT + Adapter）
│ └── utils.py # 辅助函数（分句、分词、Token 计数）
│
├── model/ # 本地模型存放目录（需自行放置）
│ ├── compression_bert_model/ # BERT 压缩模型
│ ├── paraphrase-multilingual-MiniLM-L12-v2/ # 句子编码器
│ └── sentence_labeling_adapter_cosine/ # 对话摘要 Adapter 模型
│
└── data/ # 用户上传文件临时存储（可选）


## 安装与依赖

1. **克隆或下载项目**  
   ```bash
   git clone <your-repo-url>
   cd Prompt-composer
   ```

2. **创建虚拟环境（推荐）**  
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/macOS
   .venv\Scripts\activate      # Windows
   ```

3. **安装依赖**  
   ```bash
   pip install -r requirements.txt
   ```

4. **准备模型文件**  
   - 将训练好的 BERT 压缩模型放入 `model/compression_bert_model/`  
   - 将 Sentence Transformer 模型放入 `model/paraphrase-multilingual-MiniLM-L12-v2/`  
   - 将对话摘要 Adapter 模型放入 `model/sentence_labeling_adapter_cosine/`  
   - 确保 `config.py` 中的路径指向正确位置。

## 使用说明

### 1. 启动应用

```bash
python app.py
```

浏览器访问 `http://localhost:7860` 即可使用。

### 2. 配置 LLM 后端

- 展开界面顶部的 **“模型配置（切换后端）”** 折叠面板。
- 选择后端类型：
  - **Ollama**：填写 API 地址（如 `http://localhost:11434/api/chat`）和模型名（如 `qwen2.5:3b`）。
  - **OpenAI 兼容**：填写 API 地址（如阿里云 DashScope 的 `https://dashscope.aliyuncs.com/compatible-mode/v1`）和模型名（**请使用小写**，如 `qwen-turbo`、`qwen-plus`、`qwen-max`）。如需 API Key 则填写。
- 点击 **“更新模型配置”**，状态栏显示成功即可。

### 3. 处理文档

- 上传多个 `.txt` 文件，或在下方的文本框中直接输入文本（可选）。
- 点击 **“处理文档”**，系统自动完成分句、压缩、合并、索引构建。
- 右侧“当前对话摘要”区域会显示后续的对话历史摘要。

### 4. 提问

- 在 **“请输入您的问题”** 框中输入问题。
- 通过滑块调节 **检索模式**（-1 关键词检索 → 0 平衡 → 1 语义检索）。
- 点击 **“提问”**，系统将根据检索到的相关片段和对话历史生成答案。
- 右侧 **“答案”** 区域显示结果，下方 **“Token 统计”** 显示本次调用的 Token 消耗及检索模式信息。
- 每轮问答后，**“当前对话摘要”** 会自动更新，供下一轮使用。

## 配置参数说明（`config.py`）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_PATH` | `"model/compression_bert_model"` | BERT 压缩模型路径 |
| `ENCODER_NAME` | `"model/paraphrase-multilingual-MiniLM-L12-v2"` | 句子编码器路径或 HuggingFace 模型名 |
| `SUMMARIZER_BASE` | `"model/compression_bert_model"` | 摘要模型基础 BERT 路径 |
| `SUMMARIZER_ADAPTER` | `"model/sentence_labeling_adapter_cosine/final_adapter"` | 摘要 Adapter 路径 |
| `OLLAMA_URL` | `"http://localhost:11434/api/chat"` | Ollama 默认地址 |
| `MODEL_NAME` | `"qwen2.5:3b"` | 默认模型名（Ollama） |
| `TOP_K` | `3` | 最终检索片段数 |
| `BM25_TOP_K` | `30` | 稠密检索初筛候选数 |
| `HYBRID_ALPHA` | `0.5` | 稠密检索权重（BM25 权重为 1-alpha） |
| `DOC_RELEVANCE_THRESHOLD` | `0.5` | 文档相关性阈值，低于此值跳过检索 |
| `SUMMARY_RELEVANCE_THRESHOLD` | `0.4` | 摘要相关性阈值，低于此值不加入上下文 |
| `MAX_SUMMARY_TOKENS` | `150` | 对话摘要最大字符数（近似） |

## 注意事项

- **模型大小与内存**：BERT 压缩模型和句子编码器需要一定内存，建议在 8GB+ 内存环境下运行。
- **摘要模型依赖**：确保已安装 `adapter-transformers` 且 Adapter 路径正确，否则对话摘要功能会报错。
- **API 兼容性**：使用阿里云 DashScope 时，模型名必须为小写（系统会自动转换），且 API Key 需有效。首次调用可能因模型未开通而失败，请确认已开通对应模型服务。
- **Ollama 服务**：需提前安装并运行 Ollama，拉取所需模型（如 `ollama pull qwen2.5:3b`）。
- **多用户并发**：本工具为单用户设计，多用户同时操作可能导致状态冲突，建议在个人环境使用。

## 常见问题

**Q: 上传文档后为什么没有输出统计？**  
A: 统计信息已移至底部“Token 统计”区域，处理文档时底部会显示处理完成的提示。

**Q: 如何查看当前对话摘要？**  
A: 右侧“当前对话摘要”文本框会实时显示最新生成的摘要，内容过长时可通过滚动条查看。

**Q: 调用阿里云 DashScope 返回 404**  
A: 请检查模型名是否小写（如 `qwen-max`），API Key 是否有效，以及该模型是否已开通。系统会自动转换大小写，但仍需确保模型名准确。

**Q: 可以同时使用多个文档并跨文档检索吗？**  
A: 可以。所有文档的片段会被统一索引，检索时会自动定位到相关片段，并仅扩展同一文档内的前后句，保证上下文一致。

**Q: 如何重置对话历史？**  
A: 重新点击“处理文档”即可清空历史摘要，开始全新对话。

---

如需进一步定制或遇到问题，欢迎在项目仓库提交 Issue。