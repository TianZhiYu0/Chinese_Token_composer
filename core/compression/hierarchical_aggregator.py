"""
层次化跨段聚合模块
包含段内Token聚合和跨段Transformer交互
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from typing import List, Optional

class IntraSegmentAggregator(nn.Module):
    """段内聚合：将变长Token序列压缩为固定数量的段表示向量"""
    def __init__(self, hidden_size: int = 768, num_queries: int = 4):
        super().__init__()
        self.num_queries = num_queries
        # 可学习的查询向量（类似 Perceiver 架构）
        self.queries = nn.Parameter(torch.randn(1, num_queries, hidden_size))
        # 交叉注意力层
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)

    def forward(self, token_hidden: torch.Tensor, token_mask: torch.Tensor = None):
        """
        Args:
            token_hidden: (batch, seq_len, hidden_size) 单段的Token表示
            token_mask: (batch, seq_len) 注意力掩码
        Returns:
            seg_repr: (batch, num_queries, hidden_size) 段表示向量
        """
        batch_size = token_hidden.shape[0]
        queries = self.queries.expand(batch_size, -1, -1)  # (B, Q, D)

        # 交叉注意力：查询向量 attend to Token 序列
        attn_output, _ = self.cross_attn(
            query=queries,
            key=token_hidden,
            value=token_hidden,
            key_padding_mask=~token_mask.bool() if token_mask is not None else None
        )
        queries = self.layer_norm1(queries + attn_output)

        # FFN
        ffn_output = self.ffn(queries)
        queries = self.layer_norm2(queries + ffn_output)

        return queries  # (B, Q, D)


class HierarchicalCrossSegmentAggregator(nn.Module):
    """层次化跨段聚合器"""
    def __init__(
        self,
        hidden_size: int = 768,
        num_intra_queries: int = 4,
        num_memory_tokens: int = 8,
        num_transformer_layers: int = 2,
        num_heads: int = 8
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_memory_tokens = num_memory_tokens

        # 段内聚合器（所有段落共享）
        self.intra_aggregator = IntraSegmentAggregator(hidden_size, num_intra_queries)

        # 可学习的全局记忆 Token
        self.memory_tokens = nn.Parameter(torch.randn(1, num_memory_tokens, hidden_size))

        # 跨段 Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            batch_first=True,
            activation='gelu'
        )
        self.cross_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # 可选的输出投影（将增强后的段向量映射回检索向量维度）
        self.output_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, segment_hiddens: List[torch.Tensor], segment_masks: List[torch.Tensor] = None):
        """
        Args:
            segment_hiddens: 每个段落的Token隐藏状态列表，每个形状 (1, seq_len_i, hidden_size)
            segment_masks: 对应的注意力掩码列表
        Returns:
            enhanced_segments: 增强后的段表示向量列表，每个形状 (num_intra_queries, hidden_size)
            global_memory: 全局记忆向量 (num_memory_tokens, hidden_size)
        """
        if not segment_hiddens:
            return [], self.memory_tokens.squeeze(0)

        # 第一层：段内聚合
        segment_reprs = []
        for i, seg_hidden in enumerate(segment_hiddens):
            mask = segment_masks[i] if segment_masks else None
            seg_repr = self.intra_aggregator(seg_hidden, mask)  # (1, Q, D)
            segment_reprs.append(seg_repr.squeeze(0))  # (Q, D)

        # 将所有段表示拼接为一个序列 (num_segments * Q, D)
        flat_segments = torch.cat(segment_reprs, dim=0).unsqueeze(0)  # (1, S*Q, D)

        # 第二层：拼接全局记忆 Token 并通过 Transformer
        batch_size = 1
        memory = self.memory_tokens.expand(batch_size, -1, -1)  # (1, M, D)
        full_sequence = torch.cat([memory, flat_segments], dim=1)  # (1, M + S*Q, D)

        # 跨段交互
        encoded = self.cross_transformer(full_sequence)  # (1, M + S*Q, D)

        # 分离记忆和段表示
        global_memory = encoded[:, :self.num_memory_tokens, :].squeeze(0)  # (M, D)
        enhanced_flat = encoded[:, self.num_memory_tokens:, :].squeeze(0)  # (S*Q, D)

        # 将增强后的表示重新分组为各个段落
        q_per_seg = self.intra_aggregator.num_queries
        enhanced_segments = []
        for i in range(len(segment_reprs)):
            start = i * q_per_seg
            end = start + q_per_seg
            enhanced_segments.append(enhanced_flat[start:end])  # (Q, D)

        return enhanced_segments, global_memory

    def get_document_vector(self, global_memory: torch.Tensor) -> torch.Tensor:
        """从全局记忆生成文档级检索向量（用于 FAISS 索引）"""
        # 平均池化或取第一个记忆 Token
        return global_memory.mean(dim=0)  # (hidden_size,)