"""
跨段聚合模块：对多个文档的压缩向量进行全局交互
"""
import torch
import torch.nn as nn


class CrossSegmentAggregator(nn.Module):
    def __init__(self, hidden_size=768, num_layers=2, num_heads=8, num_memory_tokens=5):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, batch_first=True, dim_feedforward=hidden_size*4
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.memory_tokens = nn.Parameter(torch.randn(1, num_memory_tokens, hidden_size))
        self.num_memory = num_memory_tokens

    def forward(self, segment_vectors):
        """
        Args:
            segment_vectors: (num_segments, hidden_size)
        Returns:
            refined_vectors: (num_segments, hidden_size) 交互后的段向量
            global_memory: (num_memory_tokens, hidden_size) 全局记忆
        """
        # 添加batch维度
        x = segment_vectors.unsqueeze(0)  # (1, num_seg, D)
        batch_size = 1
        memory = self.memory_tokens.expand(batch_size, -1, -1)  # (1, M, D)

        # 拼接记忆和段向量
        seq = torch.cat([memory, x], dim=1)  # (1, M+num_seg, D)

        # Transformer编码
        encoded = self.transformer(seq)  # (1, M+num_seg, D)

        # 分离记忆和段向量
        refined_memory = encoded[:, :self.num_memory, :].squeeze(0)  # (M, D)
        refined_segments = encoded[:, self.num_memory:, :].squeeze(0)  # (num_seg, D)

        return refined_segments, refined_memory