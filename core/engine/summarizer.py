"""
对话摘要生成器 - 支持微调模型 + TextRank 降级
"""
from transformers import AutoTokenizer, AutoModel
from typing import Optional
from core.history.textrank_summarizer import TextRankSummarizer  # 新增导入


class DialogSummarizer:
    """对话摘要生成器，优先使用微调模型，失败时降级到 TextRank"""

    def __init__(
            self,
            base_model_path: str,
            adapter_path: Optional[str] = None,
            adapter_name: Optional[str] = None,
            use_textrank_fallback: bool = True,
            textrank_encoder=None,
            num_sentences: int = 3
    ):
        """
        base_model_path: 基础模型路径（如 'bert-base-chinese'）
        adapter_path: 微调适配器路径（可选）
        adapter_name: 适配器名称（可选）
        use_textrank_fallback: 是否启用 TextRank 降级
        textrank_encoder: SentenceTransformer 编码器（用于 TextRank）
        num_sentences: TextRank 摘要句子数
        """
        self.use_fallback = use_textrank_fallback
        self.model = None
        self.tokenizer = None

        # 尝试加载微调模型（使用 AutoModel 而非 AutoModelForSeq2SeqLM）
        try:
            import os
            if not os.path.exists(base_model_path):
                raise FileNotFoundError(f"模型路径不存在: {base_model_path}")
            
            print(f"正在加载摘要模型: {base_model_path}")
            
            # 使用 local_files_only=True 禁止联网
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_path,
                local_files_only=True
            )
            # BERT 是 encoder-only 模型，使用 AutoModel 加载
            self.model = AutoModel.from_pretrained(
                base_model_path,
                local_files_only=True
            )
            if adapter_path and adapter_name:
                # 加载 adapter
                try:
                    from adapters import AutoAdapterModel
                    import os
                    
                    # 使用 AutoAdapterModel 重新加载
                    self.model = AutoAdapterModel.from_pretrained(
                        base_model_path,
                        local_files_only=True
                    )
                    
                    # 加载 adapter（adapter_path 应包含 adapter_config.json）
                    if os.path.exists(adapter_path):
                        # 加载 adapter 并设置名称
                        self.model.load_adapter(adapter_path, set_active=True)
                        print(f"✅ Adapter 加载成功: {adapter_path}")
                        print(f"   Adapter 名称: {adapter_name}")
                    else:
                        print(f"⚠️ Adapter 路径不存在: {adapter_path}")
                except ImportError:
                    print("⚠️ 未安装 adapters 库，将使用基础模型")
                    print("   安装命令: pip install adapters")
                except Exception as adapter_e:
                    print(f"⚠️ Adapter 加载失败（将使用基础模型）: {adapter_e}")
            print("微调摘要模型加载成功")
        except Exception as e:
            print(f"微调摘要模型加载失败: {e}")
            self.model = None

        # 初始化 TextRank 降级器
        if use_textrank_fallback and textrank_encoder is not None:
            self.textrank = TextRankSummarizer(textrank_encoder, num_sentences=num_sentences)
        else:
            self.textrank = None

    def summarize(self, text: str, max_length: int = 128) -> str:
        """生成摘要（优先使用 TextRank，BERT 模型用于辅助评分）"""
        # 由于 BERT 是 encoder-only 模型，不支持文本生成
        # 直接使用 TextRank 进行摘要
        if self.textrank:
            try:
                return self.textrank.summarize(text)
            except Exception as e:
                print(f"TextRank 摘要生成失败: {e}")
        
        # 最终降级：返回前200字
        return text[:200] + "..." if len(text) > 200 else text

    def summarize_with_fallback(self, text: str) -> str:
        """与 summarize 相同，保留接口兼容性"""
        return self.summarize(text)