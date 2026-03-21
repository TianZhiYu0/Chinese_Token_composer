import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from .utils import count_tokens

class HardCompressor:
    def __init__(self, model_path, device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        print("BERT 压缩模型加载成功！")

    def compress(self, sentence, keep_label_id=1):
        """
        对单个句子进行硬压缩
        返回：(compressed_text, original_tokens, compressed_tokens)
        """
        original_tokens = count_tokens(self.tokenizer, sentence)
        inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
            preds = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()
        input_ids = inputs["input_ids"].squeeze(0).cpu().numpy()
        kept_ids = [tid for tid, label in zip(input_ids, preds) if label == keep_label_id]
        compressed = self.tokenizer.decode(kept_ids, skip_special_tokens=True)
        compressed_tokens = count_tokens(self.tokenizer, compressed)
        return compressed, original_tokens, compressed_tokens