"""
词级信息优先级压缩器
使用训练好的 WordPriorityModel 进行词级压缩
"""
import torch
import jieba
from transformers import AutoTokenizer
from core.compression.word_priority_model import WordPriorityModel


class WordPriorityCompressor:
    def __init__(self, bert_model_path, priority_model_path, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_path, local_files_only=True)
        self.model = WordPriorityModel(bert_model_path).to(device)
        
        # 加载权重（兼容完整checkpoint和直接保存的state_dict）
        checkpoint = torch.load(priority_model_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            # 混合微调或其他训练保存的完整 checkpoint
            state_dict = checkpoint['model_state_dict']
        else:
            # 直接保存的 state_dict（例如原始模型）
            state_dict = checkpoint
        
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

    def compress(self, text, compression_ratio=0.7):
        """单文本压缩，返回压缩后的字符串"""
        if not text:
            return ""
        words = list(jieba.cut(text))
        if not words:
            return ""

        # 编码
        encoding = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)

        # 构建词边界
        boundaries = self._get_word_boundaries(words, encoding["input_ids"][0].tolist())

        with torch.no_grad():
            scores, _ = self.model(
                encoding["input_ids"], encoding["attention_mask"], [boundaries]
            )
        scores = scores.squeeze(0).cpu().numpy()  # (num_words,)

        # 按目标压缩比保留词
        target_keep = max(1, int(len(words) * compression_ratio))
        kept_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:target_keep]
        kept_indices.sort()  # 保持原文顺序

        compressed = "".join([words[i] for i in kept_indices])
        return compressed

    def compress_batch(self, texts, compression_ratio=0.7):
        """批量压缩，返回列表"""
        return [self.compress(t, compression_ratio) for t in texts]

    def compress_to_target_tokens(self, text, target_tokens):
        """按目标 token 数压缩（近似）"""
        # 简单按比例估算，后续可优化
        original_tokens = len(self.tokenizer.encode(text, add_special_tokens=False))
        if original_tokens <= target_tokens:
            return text
        ratio = target_tokens / original_tokens
        return self.compress(text, ratio)

    def _get_word_boundaries(self, words, input_ids):
        """计算每个词在 input_ids 中的起止位置"""
        boundaries = []
        offset = 1  # 跳过 [CLS]
        for word in words:
            word_ids = self.tokenizer.encode(word, add_special_tokens=False)
            end = offset + len(word_ids)
            boundaries.append((offset, min(end, len(input_ids))))
            offset = end
            if offset >= len(input_ids):
                break
        return boundaries