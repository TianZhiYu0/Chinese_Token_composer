"""
文档预处理器模块
================
集中管理6种文档预处理策略，为评估和问答系统提供统一的预处理接口。

预处理策略：
1. 独立分段压缩（preprocess_documents）
2. 全局压缩+片段切分（preprocess_documents_global_compress）
3. Token合并生成向量（preprocess_documents_with_merge）
4. 可学习合并+跨段聚合（preprocess_documents_with_learnable_merge）
5. 层次化跨段聚合（preprocess_documents_hierarchical）
6. 增强型层次化压缩（preprocess_documents_enhanced）
"""
import torch
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from core.compression.compressor import HardCompressor
from core.compression.merger import PromptMerger
from core.utils.utils import split_sentences, count_tokens


class DocumentPreprocessor:
    """文档预处理器：统一封装各种文档压缩与切分策略"""
    
    def __init__(self, model_path: str, encoder_name: str, device: str, 
                 compression_ratio: float = None, batch_size: int = 32,
                 compressor=None):
        """
        初始化预处理器
        
        参数:
            model_path: BERT压缩模型路径
            encoder_name: 句子编码器路径
            device: 运行设备
            compression_ratio: 默认压缩比（0.0-1.0）
            batch_size: 批处理大小
            compressor: 可选的外部压缩器实例（如 HybridCompressor, WordPriorityCompressor）
        """
        self.compressor = compressor if compressor is not None else HardCompressor(model_path, device=device)
        self.merger = PromptMerger(encoder_name, similarity_threshold=0.9)
        self.compression_ratio = compression_ratio
        self.batch_size = batch_size
        self.device = device
    
    def preprocess_independent(
        self,
        docs_dict: Dict[str, str],
        compression_ratio: float = None
    ) -> Tuple[List[str], np.ndarray, List[int], List[int]]:
        """
        策略1：独立分段压缩
        对每个文档独立进行：分句 → BERT硬压缩 → 语义合并 → 生成片段及向量
        
        参数:
            docs_dict: 文档字典 {doc_name: content}
            compression_ratio: 压缩比（0.0-1.0），None使用默认值
        
        返回:
            fragments: 压缩并合并后的文本片段列表
            vectors: 每个片段对应的稠密向量
            doc_ids: 每个片段所属的文档索引
            doc_orders: 每个片段在所属文档中的顺序编号
        """
        compression_ratio = compression_ratio or self.compression_ratio
        
        # 第一步：将所有文档切分为句子，并记录位置信息
        all_sentences = []  # 元素: (doc_idx, order, sentence_text)
        for doc_idx, content in enumerate(docs_dict.values()):
            sentences = split_sentences(content)
            for order, sent in enumerate(sentences):
                all_sentences.append((doc_idx, order, sent))
        
        if not all_sentences:
            return [], np.array([]), [], []
        
        # 第二步：批量压缩句子
        compressed_results = []
        total_batches = (len(all_sentences) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(total_batches):
            start = batch_idx * self.batch_size
            end = min(start + self.batch_size, len(all_sentences))
            batch = all_sentences[start:end]
            sentences_batch = [item[2] for item in batch]
            
            compressed_batch = self.compressor.compress_batch(
                sentences_batch,
                compression_ratio=compression_ratio
            )
            
            for (doc_idx, order, _), comp in zip(batch, compressed_batch):
                if comp:
                    compressed_results.append((doc_idx, order, comp))
        
        # 第三步：按文档重新组织压缩后的句子
        doc_sentences = defaultdict(list)
        for doc_idx, order, comp in compressed_results:
            doc_sentences[doc_idx].append((order, comp))
        
        # 第四步：对每个文档的压缩句子进行语义合并
        fragments = []
        vectors = []
        doc_ids = []
        doc_orders = []
        
        for doc_idx, sent_list in doc_sentences.items():
            sent_list.sort(key=lambda x: x[0])
            compressed_texts = [comp for _, comp in sent_list]
            
            if compressed_texts:
                result = self.merger.process(compressed_texts, do_merge=True)
                merged_frags = result['fragments']
                merged_vecs = result['vectors']
                
                for j, (frag, vec) in enumerate(zip(merged_frags, merged_vecs)):
                    fragments.append(frag)
                    vectors.append(vec)
                    doc_ids.append(doc_idx)
                    doc_orders.append(j)
        
        vectors = np.array(vectors) if vectors else np.array([])
        return fragments, vectors, doc_ids, doc_orders
    
    def preprocess_global_compress(
        self,
        docs_dict: Dict[str, str],
        compression_ratio: float = None,
        target_total_tokens: int = None,
        fragment_max_chars: int = 384
    ) -> Tuple[List[str], np.ndarray, List[int], List[int]]:
        """
        策略2：全局压缩+片段检索
        1. 合并所有文档为一个长文本
        2. 整体压缩至指定token数或按比例压缩
        3. 将压缩后文本切分为固定大小的片段
        4. 对片段进行语义合并并生成向量
        
        参数:
            docs_dict: 文档字典
            compression_ratio: 压缩比例（0-1），与 target_total_tokens 二选一
            target_total_tokens: 目标总token数，与 compression_ratio 二选一
            fragment_max_chars: 每个片段的最大字符数
        
        返回:
            fragments, vectors, doc_ids, doc_orders
        """
        compression_ratio = compression_ratio or self.compression_ratio
        
        # 1. 合并所有文档
        print("📄 正在合并所有文档...")
        all_text = "\n\n".join(docs_dict.values())
        original_tokens = count_tokens(self.compressor.tokenizer, all_text)
        print(f"   原始总 token 数: {original_tokens}")
        
        # 2. 整体压缩
        # 优先使用压缩比计算目标token数
        effective_target_tokens = None
        if compression_ratio is not None:
            effective_target_tokens = max(1, int(original_tokens * compression_ratio))
            print(f"🎯 按比例压缩，保留率 {compression_ratio:.0%}，目标 {effective_target_tokens} tokens...")
            compressed_text = self.compressor.compress(all_text, compression_ratio=compression_ratio)
        elif target_total_tokens is not None:
            print(f"🎯 目标压缩至 {target_total_tokens} tokens...")
            compressed_text = self.compressor.compress_to_target_tokens(all_text, target_total_tokens)
        else:
            compressed_text = all_text
        
        compressed_tokens = count_tokens(self.compressor.tokenizer, compressed_text)
        print(f"   压缩后 token 数: {compressed_tokens} (压缩比: {compressed_tokens/original_tokens:.2f})")
        
        # 3. 将压缩后文本切分为片段
        print(f"✂️ 正在将压缩文本切分为片段 (最大 {fragment_max_chars} 字符)...")
        sentences = split_sentences(compressed_text)
        fragments = []
        current_frag = ""
        for sent in sentences:
            if len(current_frag) + len(sent) <= fragment_max_chars:
                current_frag += sent
            else:
                if current_frag:
                    fragments.append(current_frag)
                current_frag = sent
        if current_frag:
            fragments.append(current_frag)
        print(f"   初步切分为 {len(fragments)} 个片段")
        
        # 4. 语义合并（去除高度相似的片段）
        if fragments:
            print("🔄 正在进行语义合并...")
            result = self.merger.process(fragments, do_merge=True)
            fragments = result['fragments']
            vectors = np.array(result['vectors'])
            print(f"   合并后剩余 {len(fragments)} 个片段")
        else:
            vectors = np.array([])
        
        # 片段级无文档归属概念，填充占位符
        doc_ids = list(range(len(fragments)))
        doc_orders = [0] * len(fragments)
        
        return fragments, vectors, doc_ids, doc_orders
    
    def preprocess_token_merge(
        self,
        docs_dict: Dict[str, str],
        compression_ratio: float = None,
        sim_threshold: float = 0.85,
        reduction_ratio: float = 0.3
    ) -> Tuple[List[str], np.ndarray, List[int], List[int]]:
        """
        策略3：Token合并生成向量
        使用推理时Token合并生成检索向量
        
        参数:
            docs_dict: 文档字典
            compression_ratio: 压缩比
            sim_threshold: Token合并相似度阈值
            reduction_ratio: Token合并目标减少比例
        
        返回:
            fragments, vectors, doc_ids, doc_orders
        """
        from core.compression.token_merger import create_token_merger
        
        compression_ratio = compression_ratio or self.compression_ratio
        token_merger = create_token_merger(
            self.compressor.model_path,
            device=self.device,
            sim_threshold=sim_threshold,
            reduction_ratio=reduction_ratio
        )
        
        # 第一步：分句
        all_sentences = []
        doc_names = list(docs_dict.keys())
        for doc_idx, content in enumerate(docs_dict.values()):
            sentences = split_sentences(content)
            for order, sent in enumerate(sentences):
                all_sentences.append((doc_idx, order, sent))
        
        if not all_sentences:
            return [], np.array([]), [], []
        
        # 第二步：批量压缩
        compressed_results = []
        total_batches = (len(all_sentences) + self.batch_size - 1) // self.batch_size
        for batch_idx in range(total_batches):
            start = batch_idx * self.batch_size
            end = min(start + self.batch_size, len(all_sentences))
            batch = all_sentences[start:end]
            sentences_batch = [item[2] for item in batch]
            compressed_batch = self.compressor.compress_batch(sentences_batch, compression_ratio=compression_ratio)
            for (doc_idx, order, _), comp in zip(batch, compressed_batch):
                if comp:
                    compressed_results.append((doc_idx, order, comp))
        
        # 第三步：按文档重组
        doc_sentences = defaultdict(list)
        for doc_idx, order, comp in compressed_results:
            doc_sentences[doc_idx].append((order, comp))
        
        # 第四步：语义合并生成文本片段
        fragments = []
        doc_ids_for_fragments = []
        doc_orders_for_fragments = []
        
        for doc_idx, sent_list in doc_sentences.items():
            sent_list.sort(key=lambda x: x[0])
            compressed_texts = [comp for _, comp in sent_list]
            if compressed_texts:
                result = self.merger.process(compressed_texts, do_merge=True)
                merged_frags = result['fragments']
                for j, frag in enumerate(merged_frags):
                    fragments.append(frag)
                    doc_ids_for_fragments.append(doc_idx)
                    doc_orders_for_fragments.append(j)
        
        # 第五步：为每个文档生成合并向量
        doc_vectors = []
        valid_doc_indices = []
        for doc_idx in range(len(doc_names)):
            if doc_idx not in doc_sentences:
                continue
            sent_list = doc_sentences[doc_idx]
            sent_list.sort(key=lambda x: x[0])
            doc_text = " ".join([comp for _, comp in sent_list])
            if not doc_text.strip():
                continue
            
            inputs = token_merger.tokenizer(
                doc_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                merged_hidden, merged_mask, kept_indices = token_merger(
                    inputs["input_ids"],
                    inputs["attention_mask"]
                )
            
            doc_vec = merged_hidden.mean(dim=1).squeeze(0).cpu().numpy()
            doc_vectors.append(doc_vec)
            valid_doc_indices.append(doc_idx)
        
        vectors = np.array(doc_vectors) if doc_vectors else np.array([])
        
        # 将文档向量映射回每个片段
        fragment_vectors = []
        if len(vectors) > 0:
            for doc_idx in doc_ids_for_fragments:
                try:
                    pos = valid_doc_indices.index(doc_idx)
                    fragment_vectors.append(vectors[pos])
                except ValueError:
                    fragment_vectors.append(np.zeros(vectors.shape[1]))
        else:
            fragment_vectors = []
        
        vectors = np.array(fragment_vectors) if fragment_vectors else np.array([])
        return fragments, vectors, doc_ids_for_fragments, doc_orders_for_fragments
    
    def preprocess_learnable_merge(
        self,
        docs_dict: Dict[str, str],
        compression_ratio: float = None,
        use_cross_segment: bool = True
    ) -> Tuple[List[str], np.ndarray, List[int], List[int]]:
        """
        策略4：可学习Token合并+跨段聚合
        
        参数:
            docs_dict: 文档字典
            compression_ratio: 压缩比
            use_cross_segment: 是否启用跨段聚合
        
        返回:
            fragments, vectors, doc_ids, doc_orders
        """
        from core.compression.learnable_merger import create_learnable_merger
        from core.compression.cross_segment_aggregator import CrossSegmentAggregator
        
        compression_ratio = compression_ratio or self.compression_ratio
        
        # 加载可学习合并器
        tokenizer, bert, learnable_merger = create_learnable_merger(
            self.compressor.model_path, device=self.device
        )
        
        # 加载预训练权重
        import os
        import config
        merger_weights_path = os.path.join(config.PROJECT_ROOT, "model/learnable_merger/merger.pt")
        if os.path.exists(merger_weights_path):
            learnable_merger.load_state_dict(torch.load(merger_weights_path, map_location=self.device))
            print("✅ 已加载可学习合并器权重")
        learnable_merger.eval()
        
        # 加载跨段聚合器
        aggregator = None
        if use_cross_segment:
            aggregator = CrossSegmentAggregator().to(self.device)
        
        # 第一步：分句与硬压缩
        all_sentences = []
        doc_names = list(docs_dict.keys())
        for doc_idx, content in enumerate(docs_dict.values()):
            sentences = split_sentences(content)
            for order, sent in enumerate(sentences):
                all_sentences.append((doc_idx, order, sent))
        
        if not all_sentences:
            return [], np.array([]), [], []
        
        compressed_results = []
        total_batches = (len(all_sentences) + self.batch_size - 1) // self.batch_size
        for batch_idx in range(total_batches):
            start = batch_idx * self.batch_size
            end = min(start + self.batch_size, len(all_sentences))
            batch = all_sentences[start:end]
            sentences_batch = [item[2] for item in batch]
            compressed_batch = self.compressor.compress_batch(sentences_batch, compression_ratio=compression_ratio)
            for (doc_idx, order, _), comp in zip(batch, compressed_batch):
                if comp:
                    compressed_results.append((doc_idx, order, comp))
        
        # 第二步：按文档重组
        doc_sentences = defaultdict(list)
        for doc_idx, order, comp in compressed_results:
            doc_sentences[doc_idx].append((order, comp))
        
        # 第三步：语义合并生成文本片段
        fragments = []
        doc_ids_for_fragments = []
        doc_orders_for_fragments = []
        doc_texts = {}
        
        for doc_idx, sent_list in doc_sentences.items():
            sent_list.sort(key=lambda x: x[0])
            compressed_texts = [comp for _, comp in sent_list]
            doc_text = " ".join(compressed_texts)
            doc_texts[doc_idx] = doc_text
            
            if compressed_texts:
                result = self.merger.process(compressed_texts, do_merge=True)
                merged_frags = result['fragments']
                for j, frag in enumerate(merged_frags):
                    fragments.append(frag)
                    doc_ids_for_fragments.append(doc_idx)
                    doc_orders_for_fragments.append(j)
        
        # 第四步：对每个文档应用可学习Token合并
        doc_vectors = []
        valid_doc_indices = []
        for doc_idx, doc_text in doc_texts.items():
            inputs = tokenizer(
                doc_text, return_tensors="pt", truncation=True,
                max_length=512, padding=True
            ).to(self.device)
            
            with torch.no_grad():
                bert_outputs = bert(**inputs)
                hidden_states = bert_outputs.last_hidden_state
                merged_hidden, merged_mask, _ = learnable_merger(hidden_states, inputs["attention_mask"])
                doc_vec = merged_hidden.mean(dim=1).squeeze(0).cpu().numpy()
            
            doc_vectors.append(doc_vec)
            valid_doc_indices.append(doc_idx)
        
        vectors = np.array(doc_vectors) if doc_vectors else np.array([])
        
        # 第五步：跨段聚合
        if aggregator is not None and len(vectors) > 0:
            vec_tensor = torch.tensor(vectors).to(self.device)
            refined_vecs, global_memory = aggregator(vec_tensor)
            vectors = refined_vecs.detach().cpu().numpy()
            print(f"✅ 跨段聚合完成，全局记忆维度: {global_memory.shape}")
        
        # 第六步：将文档向量映射回每个片段
        fragment_vectors = []
        if len(vectors) > 0:
            for doc_idx in doc_ids_for_fragments:
                try:
                    pos = valid_doc_indices.index(doc_idx)
                    fragment_vectors.append(vectors[pos])
                except ValueError:
                    fragment_vectors.append(np.zeros(vectors.shape[1]))
        else:
            fragment_vectors = []
        
        vectors = np.array(fragment_vectors) if fragment_vectors else np.array([])
        return fragments, vectors, doc_ids_for_fragments, doc_orders_for_fragments
    
    def preprocess_hierarchical(
        self,
        docs_dict: Dict[str, str],
        compression_ratio: float = None,
    ) -> Tuple[List[str], np.ndarray, List[int], List[int]]:
        """
        策略5：层次化跨段聚合
        
        参数:
            docs_dict: 文档字典
            compression_ratio: 压缩比
        
        返回:
            fragments, vectors, doc_ids, doc_orders
        """
        from core.compression.hierarchical_aggregator import HierarchicalCrossSegmentAggregator
        
        compression_ratio = compression_ratio or self.compression_ratio
        
        # 初始化层次化聚合器
        aggregator = HierarchicalCrossSegmentAggregator(
            hidden_size=768,
            num_intra_queries=4,
            num_memory_tokens=8,
            num_transformer_layers=2
        ).to(self.device)
        aggregator.eval()
        
        # 第一步：分句
        all_sentences = []
        for doc_idx, content in enumerate(docs_dict.values()):
            sentences = split_sentences(content)
            for order, sent in enumerate(sentences):
                all_sentences.append((doc_idx, order, sent))
        
        if not all_sentences:
            return [], np.array([]), [], []
        
        # 第二步：批量压缩句子（需HardCompressor支持return_hidden参数）
        compressed_results = []
        total_batches = (len(all_sentences) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(total_batches):
            start = batch_idx * self.batch_size
            end = min(start + self.batch_size, len(all_sentences))
            batch = all_sentences[start:end]
            sentences_batch = [item[2] for item in batch]
            
            # 注意：需要HardCompressor支持return_hidden参数
            compressed_batch, hidden_batch = self.compressor.compress_batch(
                sentences_batch,
                compression_ratio=compression_ratio,
                return_hidden=True
            )
            
            for (doc_idx, order, _), comp, hid in zip(batch, compressed_batch, hidden_batch):
                if comp:
                    compressed_results.append((doc_idx, order, comp, hid))
        
        # 第三步：按文档重组
        doc_data = defaultdict(lambda: {"orders": [], "texts": [], "hiddens": []})
        for doc_idx, order, comp, hid in compressed_results:
            doc_data[doc_idx]["orders"].append(order)
            doc_data[doc_idx]["texts"].append(comp)
            doc_data[doc_idx]["hiddens"].append(hid)
        
        # 第四步：语义合并生成文本片段
        fragments = []
        doc_ids_for_fragments = []
        doc_orders_for_fragments = []
        
        for doc_idx, data in doc_data.items():
            sorted_indices = np.argsort(data["orders"])
            texts = [data["texts"][i] for i in sorted_indices]
            if texts:
                result = self.merger.process(texts, do_merge=True)
                merged_frags = result['fragments']
                for j, frag in enumerate(merged_frags):
                    fragments.append(frag)
                    doc_ids_for_fragments.append(doc_idx)
                    doc_orders_for_fragments.append(j)
        
        # 第五步：层次化聚合生成文档向量
        doc_vectors = []
        valid_doc_indices = []
        
        for doc_idx, data in doc_data.items():
            sorted_indices = np.argsort(data["orders"])
            hiddens = [data["hiddens"][i] for i in sorted_indices]
            
            if not hiddens:
                continue
            
            segment_hiddens = [hid.to(self.device) for hid in hiddens]
            
            with torch.no_grad():
                enhanced_segs, global_mem = aggregator(segment_hiddens)
                doc_vec = aggregator.get_document_vector(global_mem)
            
            doc_vectors.append(doc_vec.cpu().numpy())
            valid_doc_indices.append(doc_idx)
        
        # 第六步：将文档向量映射回每个片段
        vectors = np.array(doc_vectors) if doc_vectors else np.array([])
        fragment_vectors = []
        if len(vectors) > 0:
            for doc_idx in doc_ids_for_fragments:
                try:
                    pos = valid_doc_indices.index(doc_idx)
                    fragment_vectors.append(vectors[pos])
                except ValueError:
                    fragment_vectors.append(np.zeros(vectors.shape[1]))
        else:
            fragment_vectors = []
        
        vectors = np.array(fragment_vectors) if fragment_vectors else np.array([])
        return fragments, vectors, doc_ids_for_fragments, doc_orders_for_fragments
    
    def preprocess_enhanced(
        self,
        docs_dict: Dict[str, str],
        question: Optional[str] = None,
        compression_ratio: float = 0.7,
    ) -> Tuple[List[str], np.ndarray, List[int], List[int]]:
        """
        策略6：增强型层次化压缩
        
        参数:
            docs_dict: 文档字典
            question: 问题（用于问题感知压缩）
            compression_ratio: 压缩比
        
        返回:
            fragments, vectors, doc_ids, doc_orders
        """
        from core.compression.enhanced_compressor import EnhancedHierarchicalCompressor
        from core.engine.summarizer import DialogSummarizer
        import config
        
        # 初始化组件
        summarizer = DialogSummarizer(
            base_model_path=config.SUMMARIZER_BASE,
            adapter_path=config.SUMMARIZER_ADAPTER,
            adapter_name=config.SUMMARIZER_ADAPTER_NAME
        )
        
        enh_compressor = EnhancedHierarchicalCompressor(
            model_path=self.compressor.model_path,
            encoder=self.merger.encoder,
            summarizer=summarizer,
            device=self.device,
            base_compression_ratio=compression_ratio,
        )
        
        documents_list = list(docs_dict.values())
        
        # 增强压缩
        compressed_docs = enh_compressor.compress_documents(documents_list, question=question)
        
        # 向量化
        fragments = compressed_docs
        vectors = []
        if fragments:
            vectors = self.merger.encoder.encode(fragments)
            vectors = np.array(vectors)
        
        doc_ids = list(range(len(fragments)))
        doc_orders = [0] * len(fragments)
        
        return fragments, vectors, doc_ids, doc_orders
