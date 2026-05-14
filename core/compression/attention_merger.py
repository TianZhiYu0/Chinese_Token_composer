"""
基于注意力的软合并适配器
========================

在 BERT 编码器之后添加可学习的合并层，通过注意力机制将 token 级别的
隐藏状态聚合为固定数量的合并 token。

核心思想：
- 冻结 BERT 参数，只训练合并层
- 使用 k 个可学习的"语义原型向量"作为查询
- 通过注意力机制聚合所有 token 的隐藏状态
- 输出固定数量的压缩 token，可直接用于下游任务

优势：
1. 不修改 BERT 内部结构
2. 压缩比精确可控（k 固定）
3. 训练稳定，收敛快
4. 支持端到端微调
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import math


class AttentionBasedMerger(nn.Module):
    """
    基于注意力的软合并层
    
    输入: BERT 输出的 token 隐藏状态 [batch, seq_len, hidden_dim]
    输出: 合并后的 token [batch, num_prototypes, hidden_dim]
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        num_prototypes: int = 64,
        dropout: float = 0.1,
        use_multi_head: bool = True,
        num_heads: int = 8
    ):
        """
        Args:
            hidden_dim: BERT 隐藏层维度（通常 768）
            num_prototypes: 原型向量数量（控制压缩比，k=64 意味着压缩到 64 个 token）
            dropout: Dropout 率
            use_multi_head: 是否使用多头注意力
            num_heads: 注意力头数
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_prototypes = num_prototypes
        self.use_multi_head = use_multi_head
        
        # 可学习的合并原型向量（核心参数）
        # 每个原型向量代表一个"语义槽"
        self.prototype_vectors = nn.Parameter(
            torch.randn(num_prototypes, hidden_dim) * math.sqrt(2.0 / hidden_dim)
        )
        
        # 多头注意力机制（可选）
        if use_multi_head:
            self.num_heads = num_heads
            assert hidden_dim % num_heads == 0, "hidden_dim 必须能被 num_heads 整除"
            self.head_dim = hidden_dim // num_heads
            
            # 查询、键、值投影
            self.q_proj = nn.Linear(hidden_dim, hidden_dim)
            self.k_proj = nn.Linear(hidden_dim, hidden_dim)
            self.v_proj = nn.Linear(hidden_dim, hidden_dim)
            self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        else:
            # 简单单头注意力
            self.q_proj = nn.Linear(hidden_dim, hidden_dim)
            self.k_proj = nn.Linear(hidden_dim, hidden_dim)
            self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Layer Normalization
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 位置编码（可选，帮助保留顺序信息）
        self.use_positional_encoding = True
        if self.use_positional_encoding:
            self.positional_encoding = self._create_positional_encoding(512, hidden_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        if self.use_multi_head:
            nn.init.xavier_uniform_(self.out_proj.weight)
    
    def _create_positional_encoding(self, max_len: int, dim: int) -> torch.Tensor:
        """创建正弦位置编码"""
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)  # [1, max_len, dim]
    
    def forward(
        self,
        token_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            token_hidden_states: BERT 输出的 token 隐藏状态 [batch, seq_len, hidden_dim]
            attention_mask: 注意力掩码 [batch, seq_len]，1 表示有效，0 表示 padding
        
        Returns:
            merged_states: 合并后的 token [batch, num_prototypes, hidden_dim]
        """
        batch_size, seq_len, _ = token_hidden_states.shape
        
        # 添加位置编码
        if self.use_positional_encoding:
            token_hidden_states = token_hidden_states + self.positional_encoding[:, :seq_len, :]
        
        # 准备原型向量（扩展到 batch）
        # prototypes: [num_prototypes, hidden_dim] -> [batch, num_prototypes, hidden_dim]
        prototypes = self.prototype_vectors.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 投影到查询、键、值空间
        # Q: 原型向量作为查询 [batch, num_prototypes, hidden_dim]
        Q = self.q_proj(prototypes)
        
        # K, V: token 隐藏状态作为键和值 [batch, seq_len, hidden_dim]
        K = self.k_proj(token_hidden_states)
        V = self.v_proj(token_hidden_states)
        
        # 多头注意力
        if self.use_multi_head:
            merged_states = self._multi_head_attention(Q, K, V, attention_mask)
        else:
            merged_states = self._single_head_attention(Q, K, V, attention_mask)
        
        # 残差连接 + LayerNorm
        merged_states = self.norm(prototypes + self.dropout(merged_states))
        
        return merged_states
    
    def _single_head_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """单头注意力"""
        # 计算注意力分数 [batch, num_prototypes, seq_len]
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.hidden_dim)
        
        # 应用注意力掩码
        if attention_mask is not None:
            # attention_mask: [batch, seq_len] -> [batch, 1, seq_len]
            mask = attention_mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax 归一化
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 加权求和
        output = torch.bmm(attention_weights, V)
        
        return self.out_proj(output)
    
    def _multi_head_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """多头注意力"""
        batch_size = Q.size(0)
        
        # 分割多头 [batch, seq_len, hidden_dim] -> [batch, seq_len, num_heads, head_dim]
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, num_heads, num_prototypes, head_dim]
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        
        # 计算注意力分数 [batch, num_heads, num_prototypes, seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 应用注意力掩码
        if attention_mask is not None:
            # [batch, 1, 1, seq_len]
            mask = attention_mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 加权求和 [batch, num_heads, num_prototypes, head_dim]
        output = torch.matmul(attention_weights, V)
        
        # 合并多头 [batch, num_prototypes, hidden_dim]
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        
        return self.out_proj(output)
    
    def get_attention_weights(
        self,
        token_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        获取注意力权重（用于可视化分析）
        
        Returns:
            attention_weights: [batch, num_heads, num_prototypes, seq_len] 或 [batch, num_prototypes, seq_len]
        """
        batch_size, seq_len, _ = token_hidden_states.shape
        prototypes = self.prototype_vectors.unsqueeze(0).expand(batch_size, -1, -1)
        
        Q = self.q_proj(prototypes)
        K = self.k_proj(token_hidden_states)
        
        if self.use_multi_head:
            Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        else:
            scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.hidden_dim)
        
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1) if not self.use_multi_head else attention_mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        return attention_weights


class SoftCompressorWithAdapter:
    """
    带适配器头的软压缩器
    
    封装 AttentionBasedMerger，提供与 HardCompressor 类似的接口
    """
    
    def __init__(
        self,
        bert_model_path: str,
        adapter_checkpoint_path: Optional[str] = None,
        num_prototypes: int = 64,
        device: Optional[str] = None,
        batch_size: int = 32
    ):
        """
        Args:
            bert_model_path: BERT 模型路径（参数冻结）
            adapter_checkpoint_path: 适配器检查点路径（可选）
            num_prototypes: 原型向量数量（控制压缩比）
            device: 运行设备
            batch_size: 批处理大小
        """
        import os
        from transformers import AutoTokenizer, AutoModel
        
        if not os.path.exists(bert_model_path):
            raise FileNotFoundError(f"BERT 模型路径不存在: {bert_model_path}")
        
        print(f"🔄 加载 BERT 编码器: {bert_model_path}")
        
        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            bert_model_path,
            local_files_only=True
        )
        self.tokenizer.model_max_length = 512
        
        # 加载 BERT 模型（冻结参数）
        self.bert_model = AutoModel.from_pretrained(
            bert_model_path,
            local_files_only=True
        )
        
        # 冻结 BERT 所有参数
        for param in self.bert_model.parameters():
            param.requires_grad = False
        
        self.bert_model.eval()
        print("✅ BERT 编码器已加载并冻结")
        
        # 获取隐藏层维度
        hidden_dim = self.bert_model.config.hidden_size
        
        # 初始化注意力合并适配器
        self.merger_adapter = AttentionBasedMerger(
            hidden_dim=hidden_dim,
            num_prototypes=num_prototypes,
            use_multi_head=True,
            num_heads=8
        )
        
        # 加载适配器检查点（如果提供）
        if adapter_checkpoint_path and os.path.exists(adapter_checkpoint_path):
            print(f"📥 加载适配器检查点: {adapter_checkpoint_path}")
            checkpoint = torch.load(adapter_checkpoint_path, map_location='cpu')
            self.merger_adapter.load_state_dict(checkpoint['adapter_state_dict'])
            print("✅ 适配器检查点加载成功")
        
        # 设置设备
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert_model.to(self.device)
        self.merger_adapter.to(self.device)
        self.batch_size = batch_size
        
        print(f"🎯 软压缩器初始化完成:")
        print(f"   - 设备: {self.device}")
        print(f"   - 原型数量: {num_prototypes}")
        print(f"   - BERT 参数: 冻结")
        print(f"   - 适配器参数: {sum(p.numel() for p in self.merger_adapter.parameters()):,} 个（可训练）")
    
    def compress(
        self,
        sentence: str,
        return_attention_weights: bool = False
    ) -> Tuple[List[float], Optional[torch.Tensor]]:
        """
        单句压缩
        
        Args:
            sentence: 输入句子
            return_attention_weights: 是否返回注意力权重（用于分析）
        
        Returns:
            compressed_embedding: 合并后的 token 嵌入 [num_prototypes, hidden_dim]
            attention_weights: 注意力权重（可选）
        """
        embeddings = self.compress_batch([sentence], return_attention_weights)
        if return_attention_weights:
            return embeddings[0][0], embeddings[0][1]
        return embeddings[0]
    
    def compress_batch(
        self,
        sentences: List[str],
        return_attention_weights: bool = False
    ) -> List[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
        批量压缩
        
        Args:
            sentences: 句子列表
            return_attention_weights: 是否返回注意力权重
        
        Returns:
            results: [(compressed_embedding, attention_weights), ...]
        """
        if not sentences:
            return []
        
        results = []
        
        # 分批处理
        for i in range(0, len(sentences), self.batch_size):
            batch_sentences = sentences[i:i + self.batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_sentences,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # BERT 编码（冻结）
            with torch.no_grad():
                outputs = self.bert_model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    output_hidden_states=True
                )
                # 使用最后一层隐藏状态
                token_hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
            
            # 注意力合并（可训练）
            self.merger_adapter.train(False)  # eval 模式
            with torch.no_grad():
                merged_states = self.merger_adapter(
                    token_hidden_states,
                    attention_mask=inputs['attention_mask']
                )  # [batch, num_prototypes, hidden_dim]
                
                if return_attention_weights:
                    attention_weights = self.merger_adapter.get_attention_weights(
                        token_hidden_states,
                        attention_mask=inputs['attention_mask']
                    )
            
            # 收集结果
            for j in range(len(batch_sentences)):
                embedding = merged_states[j].cpu()  # [num_prototypes, hidden_dim]
                attn_weights = attention_weights[j].cpu() if return_attention_weights else None
                results.append((embedding, attn_weights))
        
        return results
    
    def get_trainable_params(self) -> nn.ParameterDict:
        """获取可训练参数（用于优化器）"""
        return self.merger_adapter.state_dict()
    
    def save_adapter(self, save_path: str):
        """保存适配器检查点"""
        import os
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        checkpoint = {
            'adapter_state_dict': self.merger_adapter.state_dict(),
            'num_prototypes': self.merger_adapter.num_prototypes,
            'hidden_dim': self.merger_adapter.hidden_dim
        }
        torch.save(checkpoint, save_path)
        print(f"💾 适配器已保存到: {save_path}")
    
    def get_compression_ratio(self, original_seq_len: int) -> float:
        """计算压缩比"""
        return self.merger_adapter.num_prototypes / original_seq_len
