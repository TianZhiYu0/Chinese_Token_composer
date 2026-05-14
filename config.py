# ================= 模型路径 =================
import os
import torch

# 获取项目根目录（config.py 所在目录）
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = _PROJECT_DIR  # 暴露为全局常量

# ================= GPU 配置 =================
# 自动检测 GPU 可用性
USE_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if USE_CUDA else "cpu"
if USE_CUDA:
    print(f"✅ GPU 可用: {torch.cuda.get_device_name(0)}")
    print(f"   GPU 数量: {torch.cuda.device_count()}")
    print(f"   CUDA 版本: {torch.version.cuda}")
else:
    print("⚠️ 未检测到 GPU，将使用 CPU 运行")

# BERT 硬压缩模型（用于 token 级压缩）
MODEL_PATH = os.path.join(_PROJECT_DIR, "model", "compression_bert_mooscomp_news")

# 硬压缩比配置（0-1之间，None表示使用模型原始预测）
# 示例：0.5 = 保留50% token（压缩50%），0.7 = 保留70%（压缩30%）
HARD_COMPRESSION_RATIO = 0.7  # 默认None，使用原始策略；可设置为 0.5-0.8 之间的值

# 句子编码器（用于语义合并和稠密检索）
ENCODER_NAME = os.path.join(_PROJECT_DIR, "model", "paraphrase-multilingual-MiniLM-L12-v2")

# 对话摘要 Adapter 模型（用于历史压缩）
SUMMARIZER_BASE = os.path.join(_PROJECT_DIR, "model", "compression_bert_model")
SUMMARIZER_ADAPTER = os.path.join(_PROJECT_DIR, "model", "sentence_labeling_adapter_cosine", "final_adapter")
SUMMARIZER_ADAPTER_NAME = "key_sentence_labeling"

# ================= Ollama 配置 =================
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:3b-32k"      # 自定义模型（32K上下文窗口）

# ================= 检索参数 =================
SIMILARITY_THRESHOLD = 0.9         # 语义合并阈值（暂未使用）
TOP_K = 5                         # 原为 5，增加召回片段数
HYBRID_ALPHA = 0.5                # 原为 0.5，改为负值使 BM25 权重 = (1 - (alpha+1)/2) = 0.9
BM25_TOP_K = 50                   # 原为 50，增加初筛候选数
RELEVANCE_THRESHOLD = 0.4          # 原为 0.5，略微放宽过滤
BATCH_SIZE = 32   # 可根据显存调整

# 词级优先级压缩模型
WORD_PRIORITY_BERT_PATH = os.path.join(_PROJECT_DIR, "model", "compression_bert_mooscomp_news")
WORD_PRIORITY_MODEL_PATH = os.path.join(_PROJECT_DIR, "model", "word_priority_model_mixed", "best_model.pt")

# 混合压缩模型 (Hybrid: MOOSComp + WordPriority)
HYBRID_WORD_PRIORITY_MODEL_PATH = os.path.join(_PROJECT_DIR, "model", "word_priority_model_mixed", "best_model.pt")

# 基座 BERT# ================= 对话历史压缩参数 =================
HISTORY_COMPRESS_THRESHOLD = 5     # 超过此轮数触发压缩
COMPRESS_WINDOW = 5                # 每次压缩最近的 N 轮
MAX_SUMMARY_TOKENS = 500           # 摘要最大 token 数（估算）
# 对话摘要历史长度限制（字符数，滑动窗口）
MAX_HISTORY_LEN = 500
# 相关性阈值
DOC_RELEVANCE_THRESHOLD = 0.5        # 文档相关性阈值，低于此值跳过检索
SUMMARY_RELEVANCE_THRESHOLD = 0.4    # 摘要相关性阈值，低于此值不加入上下文

# 历史语境构建参数
SELECTED_SUMMARIES_COUNT = 3        # 选取最相关的历史摘要数量
MAX_CONTEXT_TOKENS = 500            # 历史语境最大 Token 数（支持3-5轮对话）
CONTEXT_WINDOW_SIZE = 4096          # 模型上下文窗口大小 (tokens) 优化：

# 上下文构建优化参数
TOKEN_ELASTICITY = 0.1              # Token 预算允许 10% 浮动
MIN_SENTENCE_TOKENS = 8             # 截断时最小保留句子长度
DENSITY_WEIGHT = 0.3                # 信息密度在排序中的权重
MAX_CHUNKS_PER_DOC = 2              # 每篇文档最多选取片段数
CHUNK_SIMILARITY_THRESHOLD = 0.7    # 重复过滤阈值（Rouge-L 相似度）

# 摘要查重阈值（余弦相似度高于此值视为重复）
SUMMARY_DEDUP_THRESHOLD = 0.85

# ================= 检索融合模型 (T5) =================
CHUNK_FUSER_MODEL = os.path.join(_PROJECT_DIR, "model", "mengzi-t5-finetuned-fusion")
USE_CHUNK_FUSION = True  # 已训练完成，启用融合
CHUNK_FUSION_THRESHOLD = 0.75  # 融合分组阈值（相似度高于此值进行融合）

# ================= 软压缩模块 (Soft Compression) =================
USE_SOFT_COMPRESSION = False  # 默认关闭，需要 T5 解码器支持
SOFT_COMPRESSION_BERT = os.path.join(_PROJECT_DIR, "model", "compression_bert_model")
SOFT_COMPRESSION_ADAPTER = os.path.join(_PROJECT_DIR, "model", "attention_merger_adapter", "final_adapter.pt")
T5_DECODER_CHECKPOINT = os.path.join(_PROJECT_DIR, "model", "t5_decoder_lora", "best_decoder.pt")
T5_DECODER_BASE = os.path.join(_PROJECT_DIR, "model", "mengzi-t5-base")  # T5 基础模型

# 自适应压缩阈值：文档总token数 <= 窗口 * 此倍数时，启用全文压缩直答模式
FULL_COMPRESSION_THRESHOLD_MULTIPLIER = 10

# 关键词检索模式开关（在压缩检索增强模式下使用）
USE_KEYWORD_RETRIEVAL = True
KEYWORD_RETRIEVAL_TOP_K = 5
GLOBAL_COMPRESS_BEFORE_CHUNK =True  # 设为 True 启用新模式

# Web界面默认参数
FRAGMENT_SIZE = 384  # 全局压缩模式下每个片段的大小（字符数）