import torch
from transformers import BertTokenizer
from adapters import BertAdapterModel
import numpy as np
import re
from .utils import chinese_word_seg

class DialogSummarizer:
    """
    基于 BERT + Adapter 的对话摘要模型
    使用训练好的 Adapter 对对话进行压缩/摘要
    """
    def __init__(self, base_model_path, adapter_path, adapter_name="key_sentence_labeling", device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(base_model_path)
        self.model = BertAdapterModel.from_pretrained(base_model_path)
        self.model.load_adapter(adapter_path, load_as=adapter_name)
        self.model.set_active_adapters(adapter_name)
        self.model.to(self.device)
        self.model.eval()
        print("对话摘要模型加载成功！")

    def _split_sentences(self, text):
        """简单分句，用于摘要抽取"""
        if not text:
            return []
        sentences = re.split(r'([。！？\n])', text)
        result = []
        for i in range(0, len(sentences)-1, 2):
            sent = sentences[i] + sentences[i+1]
            sent = sent.strip()
            if sent and len(sent) > 5:
                result.append(sent)
        if len(sentences) % 2 == 1:
            sent = sentences[-1].strip()
            if sent and len(sent) > 5:
                result.append(sent)
        return result

    def _predict_score(self, sentence):
        """返回句子的重要性分数"""
        # 添加截断
        inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            if len(probs) > 2:
                token_probs = probs[1:-1]
            else:
                token_probs = probs
            # 取最大概率作为句子分数
            score = np.max(token_probs[:, 1]) if len(token_probs) > 0 else 0.0
        return score

    def summarize(self, dialog_text, max_length=150):
        """
        生成对话摘要
        dialog_text: 字符串，格式如 "Human: ...\nAssistant: ..."
        max_length: 摘要最大字符数（近似）
        返回摘要字符串
        """
        # 将对话按轮次分割（简单分割）
        rounds = re.split(r'\n(?=Human:|Assistant:)', dialog_text)
        summary_parts = []

        for turn in rounds:
            turn = turn.strip()
            if not turn:
                continue
            if turn.startswith("Human:"):
                summary_parts.append(turn)
            elif turn.startswith("Assistant:"):
                text = turn[10:].strip()
                sentences = self._split_sentences(text)
                if not sentences:
                    continue
                scores = [self._predict_score(s) for s in sentences]
                paired = list(zip(sentences, scores))
                paired.sort(key=lambda x: x[1], reverse=True)
                k = max(1, min(3, len(sentences) // 2))
                selected = [s for s, _ in paired[:k]]
                order = {s: i for i, s in enumerate(sentences)}
                selected.sort(key=lambda x: order[x])
                summary_parts.append(f"Assistant: {' '.join(selected)}")
            else:
                summary_parts.append(turn)

        summary = "\n".join(summary_parts)
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."
        return summary