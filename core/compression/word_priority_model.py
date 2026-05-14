"""
词级信息优先级回归模型
基座：compression_bert_mooscomp_news
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import jieba


class WordPriorityModel(nn.Module):
    def __init__(self, bert_model_path: str, hidden_size: int = 768, dropout: float = 0.1):
        super().__init__()
        # 加载预训练的 BERT 编码器（冻结底层，仅微调顶层）
        self.bert = AutoModel.from_pretrained(bert_model_path, local_files_only=True)

        # 上下文建模（轻量级 BiLSTM，帮助捕捉词语间的依赖关系）
        self.context_encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        # 回归头：将上下文表示映射为单一优先级分数
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()  # 输出 0~1
        )

    def forward(self, input_ids, attention_mask, word_boundaries):
        """
        input_ids: (batch, seq_len)
        attention_mask: (batch, seq_len)
        word_boundaries: 每个样本的词语边界列表，格式为 [(start, end), ...]

        返回: (batch, max_words) 每个词的优先级分数
        """
        batch_size = input_ids.size(0)

        # 1. BERT 编码
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_hidden = outputs.last_hidden_state  # (batch, seq_len, 768)

        # 2. 词级池化：对每个词内的 Token 取平均
        word_vectors_list = []
        word_masks_list = []
        max_words = 0

        for b in range(batch_size):
            boundaries = word_boundaries[b]  # [(0,2), (3,5), ...]
            word_vecs = []
            for start, end in boundaries:
                if start < end and start < token_hidden.size(1):
                    vec = token_hidden[b, start:min(end, token_hidden.size(1)), :].mean(dim=0)
                    word_vecs.append(vec)
                else:
                    word_vecs.append(torch.zeros(token_hidden.size(-1), device=token_hidden.device))

            word_vectors_list.append(torch.stack(word_vecs))
            max_words = max(max_words, len(boundaries))

        # 填充到相同长度
        word_vectors_padded = torch.zeros(batch_size, max_words, token_hidden.size(-1), device=token_hidden.device)
        word_masks = torch.zeros(batch_size, max_words, device=token_hidden.device)
        for b in range(batch_size):
            n_words = word_vectors_list[b].size(0)
            word_vectors_padded[b, :n_words, :] = word_vectors_list[b]
            word_masks[b, :n_words] = 1.0

        # 3. 上下文建模
        context_output, _ = self.context_encoder(word_vectors_padded)  # (batch, max_words, 768)

        # 4. 预测分数
        scores = self.regressor(context_output).squeeze(-1)  # (batch, max_words)
        scores = scores * word_masks  # 掩码掉填充位置

        return scores, word_masks