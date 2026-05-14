# core/compression/compression_strategy.py
"""
压缩策略控制器
===============
根据文档长度和用户参数自适应选择最优压缩与检索策略。

策略选择逻辑：
1. 如果显式指定 --global_compress,强制使用全局压缩+片段检索
2. 如果文档总token数 <= 窗口大小，使用全文压缩直答（优先选择）
3. 如果文档总token数 <= 阈值（窗口大小*10），使用全文压缩直答
4. 否则使用标准压缩检索增强模式

支持的压缩策略详解：

┌─────────────────────────────────────────────────────────────────────┐
│ 策略名称 │ 模式标识 │ 适用场景 │ 核心特点 │
├─────────────────────────────────────────────────────────────────────┤
│ 全文压缩直答 │ full_compress │ 文档较短（<=20000 tokens） │ 直接压缩后送入LLM │
│ 全局压缩+检索 │ global_compress_retrieve │ 需全局理解的场景 │ 先合并压缩再片段化 │
│ 独立分段压缩 │ compress_retrieve → independent │ 多文档独立处理 │ 每文档独立压缩合并 │
│ Token合并检索 │ compress_retrieve → token_merge │ 需保留Token级信息 │ 推理时Token级合并 │
│ 可学习合并检索 │ compress_retrieve → learnable_merge │ 复杂语义关系 │ 可学习的合并权重 │
└─────────────────────────────────────────────────────────────────────┘

各策略详细说明：

1. 全文压缩直答 (full_compress)
   - 适用：文档总token数 <= 20000
   - 流程：合并所有文档 → 整体压缩 → 直接送入LLM
   - 优点：保留全局上下文，回答更连贯
   - 缺点：受限于LLM上下文窗口

2. 全局压缩+片段检索 (global_compress_retrieve)
   - 适用：需要全局理解但文档较长
   - 流程：合并所有文档 → 整体压缩 → 切分为片段 → 语义合并
   - 优点：保持全局语义一致性
   - 缺点：文档边界信息丢失

3. 独立分段压缩 (independent)
   - 适用：多文档独立处理场景
   - 流程：分句 → 独立压缩 → 按文档语义合并 → 生成片段
   - 优点：保留文档边界，适合多文档问答
   - 缺点：可能丢失跨文档关联

4. Token合并检索 (token_merge)
   - 适用：需要保留Token级精细信息
   - 流程：分句 → 压缩 → 推理时Token级语义合并
   - 优点：保留更多细节信息
   - 缺点：计算复杂度较高

5. 可学习合并检索 (learnable_merge)
   - 适用：复杂语义关系场景
   - 流程：使用训练好的合并模型进行可学习的Token合并
   - 优点：自适应学习最佳合并策略
   - 缺点：需要额外训练，推理较慢
"""
import os
import numpy as np
from typing import Dict, List, Tuple, Optional
import config
from core.compression.preprocessors import DocumentPreprocessor
from core.compression.compressor import HardCompressor
from core.utils.utils import count_tokens


class CompressionStrategy:
    """自适应压缩策略控制器"""
    
    def __init__(self, documents: Dict[str, str]):
        """
        初始化策略控制器
        
        参数:
            documents: 文档字典 {doc_name: content}
        """
        self.documents = documents
        self.full_text = "\n\n".join(documents.values())
        self.temp_compressor = HardCompressor(config.MODEL_PATH, device=config.DEVICE)
        self.total_tokens = count_tokens(self.temp_compressor.tokenizer, self.full_text)
        
        # 创建预处理器实例
        self.preprocessor = DocumentPreprocessor(
            model_path=config.MODEL_PATH,
            encoder_name=config.ENCODER_NAME,
            device=config.DEVICE,
            batch_size=getattr(config, 'BATCH_SIZE', 32)
        )
    
    def select_mode(self, args) -> str:
        """
        根据参数和文档长度返回模式字符串

        参数:
            args: 命令行参数

        返回:
            模式字符串：'global_compress_retrieve' / 'full_compress' / 'compress_retrieve' / 'vanilla'
        """
        if args.mode != "rag":
            return "vanilla"

        # 优先级1：显式指定全局压缩
        if args.global_compress:
            return "global_compress_retrieve"

        # 优先级2：当使用基于窗口的自动压缩时，默认使用全文压缩
        # 只有当文档token数超过窗口大小时，才考虑使用片段检索模式
        if self.total_tokens <= config.CONTEXT_WINDOW_SIZE:
            return "full_compress"

        # 优先级3：根据文档长度自适应（仅当token数超过窗口时才使用片段检索）
        threshold = config.CONTEXT_WINDOW_SIZE * config.FULL_COMPRESSION_THRESHOLD_MULTIPLIER
        if self.total_tokens <= threshold:
            return "full_compress"
        else:
            return "compress_retrieve"
    
    def execute_preprocess(self, mode: str, args) -> Tuple[List[str], np.ndarray, List[int], List[int]]:
        """
        根据模式调用对应的预处理策略
        
        参数:
            mode: 模式字符串
            args: 命令行参数
        
        返回:
            fragments, vectors, doc_ids, doc_orders
        """
        if mode == "global_compress_retrieve":
            print(f"🌐 [全局压缩+检索模式] 开始预处理文档...")
            print(f"   说明：将所有文档合并后整体压缩，再切分为片段进行检索")
            if args.compression_ratio is not None:
                print(f"🎯 [压缩比配置] 使用动态压缩比: {args.compression_ratio:.0%}")
            
            # 目标总 token 数设定为窗口大小（与直答模式公平对比）
            target_tokens = config.CONTEXT_WINDOW_SIZE
            return self.preprocessor.preprocess_global_compress(
                self.documents,
                compression_ratio=args.compression_ratio,
                target_total_tokens=target_tokens,
                fragment_max_chars=args.fragment_size
            )
        
        elif mode == "compress_retrieve":
            print(f"📄 [RAG模式] 开始预处理文档...")
            print(f"   说明：对每个文档独立压缩后进行语义合并")
            if args.compression_ratio is not None:
                print(f"🎯 [压缩比配置] 使用动态压缩比: {args.compression_ratio:.0%}")
            
            # 根据参数选择预处理策略
            if args.use_learnable_merge:
                print(f"   策略：可学习Token合并 + 跨段聚合")
                return self.preprocessor.preprocess_learnable_merge(
                    self.documents,
                    compression_ratio=args.compression_ratio,
                    use_cross_segment=args.use_cross_segment
                )
            elif args.use_token_merge:
                print(f"   策略：推理时Token合并")
                return self.preprocessor.preprocess_token_merge(
                    self.documents,
                    compression_ratio=args.compression_ratio,
                    sim_threshold=args.merge_sim_threshold,
                    reduction_ratio=args.merge_reduction_ratio
                )
            else:
                print(f"   策略：独立分段压缩")
                return self.preprocessor.preprocess_independent(
                    self.documents,
                    compression_ratio=args.compression_ratio
                )
        
        else:
            # full_compress 或 vanilla 不需要预处理返回片段
            if mode == "full_compress":
                print(f"📝 [全文压缩直答模式] 文档较短，直接压缩后送入LLM")
            return [], np.array([]), [], []
