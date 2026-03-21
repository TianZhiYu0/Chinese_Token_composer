# ================= 模型路径 =================
# BERT 硬压缩模型（用于 token 级压缩）
MODEL_PATH = "model/compression_bert_model"

# 句子编码器（用于语义合并和稠密检索）
ENCODER_NAME = "model/paraphrase-multilingual-MiniLM-L12-v2"

# 对话摘要 Adapter 模型（用于历史压缩）
SUMMARIZER_BASE = "model/compression_bert_model"          # 基础 BERT 模型
SUMMARIZER_ADAPTER = "model/sentence_labeling_adapter_cosine/final_adapter"  # Adapter 路径
SUMMARIZER_ADAPTER_NAME = "key_sentence_labeling"

# ================= Ollama 配置 =================
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:3b"          # 可根据实际情况修改

# ================= 检索参数 =================
SIMILARITY_THRESHOLD = 0.8         # 语义合并阈值（暂未使用）
TOP_K = 3                          # 最终检索片段数
HYBRID_ALPHA = 0.5                 # 稠密检索权重（BM25 权重为 1-alpha）
BM25_TOP_K = 30                    # 稠密检索初筛候选数（用于 BM25 融合）
RELEVANCE_THRESHOLD = 0.5   # 话题相关性阈值，低于此值则跳过文档检索
# ================= 对话历史压缩参数 =================
HISTORY_COMPRESS_THRESHOLD = 5     # 超过此轮数触发压缩
COMPRESS_WINDOW = 5                # 每次压缩最近的 N 轮
MAX_SUMMARY_TOKENS = 150           # 摘要最大 token 数（估算）
# 相关性阈值
DOC_RELEVANCE_THRESHOLD = 0.5        # 文档相关性阈值，低于此值跳过检索
SUMMARY_RELEVANCE_THRESHOLD = 0.4    # 摘要相关性阈值，低于此值不加入上下文