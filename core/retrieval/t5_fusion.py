"""
T5 智能文本融合器
=================
使用训练好的 T5 模型对高相似度文本进行语义融合，而非简单拼接

功能：
1. 检测高相似度文本对
2. 使用 T5 生成融合后的文本
3. 保留关键信息，去除冗余
"""
import torch
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, T5ForConditionalGeneration
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class T5FusionEngine:
    """T5 文本融合引擎"""
    
    def __init__(
        self, 
        model_path: str = "model/mengzi-t5-finetuned-fusion",
        device: str = None,
        use_4bit: bool = False,
        freeze_params: bool = True  # 新增：是否冻结参数（默认冻结）
    ):
        """
        Args:
            model_path: T5 模型路径（支持完整模型或 LoRA Adapter）
            device: 运行设备 (cpu/cuda)
            use_4bit: 是否使用4bit量化
            freeze_params: 是否冻结模型参数（推理模式默认True，训练模式设False）
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fusion_threshold = 0.75  # 融合阈值
        self.max_input_length = 512
        self.max_output_length = 256
        
        print(f"🔄 加载 T5 融合模型: {model_path}")
        print(f"   设备: {self.device}")
        
        # 检查是否为 LoRA Adapter 格式
        import os
        is_lora_adapter = os.path.exists(os.path.join(model_path, "adapter_config.json"))
        
        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True
        )
        
        # 加载模型
        if is_lora_adapter:
            # LoRA Adapter 模式：需要加载基础模型 + Adapter
            from peft import PeftModel
            
            # 推断基础模型路径（支持多种目录结构）
            base_model_name = "mengzi-t5-base"
            
            # 尝试多种可能的路径
            possible_paths = [
                os.path.join(os.path.dirname(model_path), base_model_name),  # model/mengzi-t5-base
                os.path.join(os.path.dirname(model_path), base_model_name, "langboat", base_model_name),  # ModelScope 嵌套
                base_model_name,  # 直接使用模型名（会从 HuggingFace 下载）
            ]
            
            base_model_path = None
            for path in possible_paths:
                if os.path.exists(path) and os.path.exists(os.path.join(path, "config.json")):
                    base_model_path = path
                    break
            
            if base_model_path is None:
                # 如果都找不到，使用第一个路径（会报错提示）
                base_model_path = possible_paths[0]
            
            print(f"   检测到 LoRA Adapter 格式")
            print(f"   基础模型: {base_model_path}")
            
            # 加载基础模型
            base_model = T5ForConditionalGeneration.from_pretrained(
                base_model_path,
                local_files_only=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            # 加载 LoRA Adapter
            self.model = PeftModel.from_pretrained(base_model, model_path)
        elif use_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            self.model = T5ForConditionalGeneration.from_pretrained(
                model_path,
                local_files_only=True,
                quantization_config=quantization_config,
                device_map="auto"
            )
        else:
            self.model = T5ForConditionalGeneration.from_pretrained(
                model_path,
                local_files_only=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            self.model = self.model.to(self.device)
        
        self.model.eval()
        
        # 冻结模型参数（根据 freeze_params 参数决定）
        if freeze_params:
            frozen_count = 0
            total_params = 0
            for param in self.model.parameters():
                param.requires_grad = False
                frozen_count += 1
                total_params += param.numel()
            
            print("✅ T5 融合模型加载完成")
            print(f"   参数已冻结（推理模式）")
            print(f"   冻结参数数量: {frozen_count} 个张量, {total_params:,} 个参数")
        else:
            print("✅ T5 融合模型加载完成")
            print(f"   参数未冻结（训练模式，可微调）")
    
    def build_fusion_prompt(self, text1: str, text2: str) -> str:
        """
        构建融合提示词（优化版，强调去重和保留关键信息）
        
        Args:
            text1: 文本1
            text2: 文本2
            
        Returns:
            融合提示词
        """
        prompt = (
            f"融合以下两段文本，要求："
            f"1. 删除重复信息"
            f"2. 保留所有关键数字、时间、人名"
            f"3. 语言简洁流畅"
            f"\n\n"
            f"文本1：{text1}"
            f"\n"
            f"文本2：{text2}"
            f"\n\n"
            f"融合结果："
        )
        return prompt
    
    def fuse_texts(self, text1: str, text2: str, max_length: int = 256) -> str:
        """
        使用 T5 模型进行真正的文本融合生成
        
        Args:
            text1: 文本1
            text2: 文本2
            max_length: 最大输出长度
            
        Returns:
            融合后的文本
        """
        try:
            # 构建融合提示词
            prompt = self.build_fusion_prompt(text1, text2)
            
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_input_length
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # T5 生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=5,              # 增加 beam 数量
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    length_penalty=0.6,       # 更鼓励简洁（之前 0.8）
                    repetition_penalty=1.2    # 惩罚重复内容
                )
            
            # 解码
            fused_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 后处理：确保关键信息保留
            fused_text = self._post_process_fused_text(fused_text, text1, text2)
            
            return fused_text
            
        except Exception as e:
            print(f"  ⚠️ T5 融合失败，回退到规则融合: {e}")
            # 如果 T5 生成失败，回退到规则融合
            return self._smart_merge(text1, text2)
    
    def _post_process_fused_text(self, fused_text: str, text1: str, text2: str) -> str:
        """
        后处理融合文本，确保关键信息保留
        
        Args:
            fused_text: 融合后的文本
            text1: 原文1
            text2: 原文2
            
        Returns:
            清理后的融合文本
        """
        import re
        
        # 1. 清理特殊符号
        fused_text = re.sub(r'^[:：,，;；]+', '', fused_text)  # 去除开头的标点
        fused_text = re.sub(r'\s+', ' ', fused_text)  # 多个空格变一个
        
        # 2. 提取关键信息（数字、金额、人名、时间）
        original_text = text1 + " " + text2
        key_patterns = [
            r'\d{4}年',         # 年份 (如 2024年)
            r'\d+(?:\.\d+)?%',  # 百分比 (如 5.2%, 6.3%)
            r'\d+\.\d+',        # 小数 (如 5.2, 6.3)
            r'[\u4e00-\u9fa5]{2,4}(同志|主席|部长)',  # 人名+职务
            r'\d+[万亿]?元',    # 金额（带单位）(如 100亿元, 126万亿元)
        ]
        
        key_infos = []
        for pattern in key_patterns:
            matches = re.findall(pattern, original_text)
            # 对于分组捕获，取完整匹配
            if matches and isinstance(matches[0], tuple):
                key_infos.extend([m[0] if isinstance(m, tuple) else m for m in re.finditer(pattern, original_text)])
            else:
                key_infos.extend(matches)
        
        # 3. 过滤掉非关键信息（称谓词）
        non_critical = {'同志', '先生', '女士', '小姐'}
        key_infos = [k for k in key_infos if k not in non_critical]
        
        # 4. 检查关键信息是否保留
        missing_keys = []
        for key in key_infos:
            if key not in fused_text:
                # 特殊处理：检查是否是数字格式变化（如 39,218 vs 39218）
                key_normalized = re.sub(r'[,，]', '', key)
                fused_normalized = re.sub(r'[,，]', '', fused_text)
                if key_normalized not in fused_normalized:
                    missing_keys.append(key)
        
        # 5. 检测幻觉：如果输出长度超过原文总和的 1.2 倍，判定为幻觉
        max_expected_length = (len(text1) + len(text2)) * 1.2
        
        # 6. 如果丢失关键信息或过度膨胀，回退到规则融合
        if len(fused_text) > max_expected_length or len(missing_keys) > 0:
            if missing_keys:
                print(f"    ⚠️ 关键信息丢失: {missing_keys}")
            # 简单去重拼接策略
            return self._smart_merge(text1, text2)
        
        return fused_text
    
    def _smart_merge(self, text1: str, text2: str) -> str:
        """
        智能融合策略（基于规则和语义去重）
        
        策略：
        1. 提取关键信息（数字、时间、人名）
        2. 按句子分割
        3. 去除重复/相似句子（使用包含关系）
        4. 智能拼接，保留所有关键信息
        """
        import re
        
        # 1. 提取关键信息（用于验证）
        original_text = text1 + " " + text2
        key_patterns = [
            r'\d{4}年',  # 年份
            r'\d+%',     # 百分比
            r'\d+\.\d+', # 小数
            r'[\u4e00-\u9fa5]{2,4}(同志|主席|部长|主席)',  # 人名+职务
        ]
        
        key_infos = []
        for pattern in key_patterns:
            key_infos.extend(re.findall(pattern, original_text))
        
        # 2. 按句子分割
        sentences1 = re.split(r'[。；;！!]', text1)
        sentences2 = re.split(r'[。；;！!]', text2)
        
        # 清理空句子
        sentences1 = [s.strip() for s in sentences1 if s.strip()]
        sentences2 = [s.strip() for s in sentences2 if s.strip()]
        
        # 3. 去重合并（使用包含关系判断）
        merged_sentences = []
        
        # 先添加text1的所有句子
        for s in sentences1:
            if s:
                # 检查是否已有更完整的版本
                is_duplicate = False
                for existing in merged_sentences:
                    # 如果s被existing包含，或者existing被s包含，视为重复
                    if s in existing or existing in s:
                        # 保留更长的（信息更完整）
                        if len(s) > len(existing):
                            merged_sentences.remove(existing)
                            merged_sentences.append(s)
                        is_duplicate = True
                        break
                    # 或者使用关键词重叠判断
                    elif self._sentence_similarity(s, existing) >= 0.4:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    merged_sentences.append(s)
        
        # 再添加text2的不重复句子
        for s in sentences2:
            if s:
                is_duplicate = False
                for existing in merged_sentences:
                    if s in existing or existing in s:
                        if len(s) > len(existing):
                            merged_sentences.remove(existing)
                            merged_sentences.append(s)
                        is_duplicate = True
                        break
                    elif self._sentence_similarity(s, existing) >= 0.4:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    merged_sentences.append(s)
        
        # 4. 拼接结果
        result = '。'.join(merged_sentences) + '。'
        
        return result
    
    def _is_similar_to_any(self, sentence: str, sentences: list, threshold: float = 0.8) -> bool:
        """检查句子是否与列表中任意句子相似"""
        for existing in sentences:
            if self._sentence_similarity(sentence, existing) >= threshold:
                return True
        return False
    
    def _sentence_similarity(self, s1: str, s2: str) -> float:
        """
        计算句子相似度（改进版，考虑关键信息）
        """
        import re
        
        # 提取关键信息
        def extract_keys(s):
            keys = set()
            # 数字
            keys.update(re.findall(r'\d+\.?\d*', s))
            # 中文词汇
            keys.update(re.findall(r'[\u4e00-\u9fa5]{2,}', s))
            return keys
        
        keys1 = extract_keys(s1)
        keys2 = extract_keys(s2)
        
        if not keys1 or not keys2:
            return 0.0
        
        # Jaccard 相似度
        intersection = keys1 & keys2
        union = keys1 | keys2
        
        return len(intersection) / len(union)
    
    def find_similar_pairs(
        self, 
        chunks: List[Dict], 
        threshold: float = 0.75
    ) -> List[Tuple[int, int, float]]:
        """
        找出所有相似的文本对
        
        Args:
            chunks: 文本块列表 [{"content": str, ...}, ...]
            threshold: 相似度阈值
            
        Returns:
            [(idx1, idx2, similarity), ...]
        """
        if len(chunks) < 2:
            return []
        
        # 编码所有文本
        texts = [c["content"] for c in chunks]
        
        # 使用 encoder 计算相似度（如果有）
        if hasattr(self, 'encoder'):
            embs = self.encoder.encode(texts)
            sim_matrix = cosine_similarity(embs)
        else:
            #  fallback: 使用简单的TF-IDF
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(texts)
            sim_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
        
        # 找出相似对
        pairs = []
        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                sim = sim_matrix[i][j]
                if sim >= threshold:
                    pairs.append((i, j, sim))
        
        # 按相似度降序排序
        pairs.sort(key=lambda x: x[2], reverse=True)
        
        return pairs
    
    def deduplicate_with_fusion(
        self, 
        chunks: List[Dict], 
        threshold: float = 0.75,
        enable_fusion: bool = True
    ) -> List[Dict]:
        """
        使用 T5 融合去重
        
        策略：
        1. 找出相似文本对
        2. 对高相似度文本使用 T5 融合
        3. 保留不相似的文本
        
        Args:
            chunks: 文本块列表
            threshold: 相似度阈值
            enable_fusion: 是否启用 T5 融合（否则使用简单策略）
            
        Returns:
            去重后的文本块列表
        """
        if len(chunks) <= 1:
            return chunks
        
        print(f"  [T5融合] 开始去重，原始 {len(chunks)} 个文本块")
        
        # 找出相似对
        similar_pairs = self.find_similar_pairs(chunks, threshold)
        
        if not similar_pairs:
            print(f"  [T5融合] 未发现相似文本对，返回原始结果")
            return chunks
        
        print(f"  [T5融合] 发现 {len(similar_pairs)} 对相似文本")
        
        # 追踪哪些索引已被融合
        merged_indices = set()
        fused_chunks = []
        
        for idx1, idx2, sim_score in similar_pairs:
            # 跳过已融合的
            if idx1 in merged_indices or idx2 in merged_indices:
                continue
            
            chunk1 = chunks[idx1]
            chunk2 = chunks[idx2]
            
            if enable_fusion:
                # 使用 T5 融合
                print(f"  [T5融合] 融合文本对 (相似度: {sim_score:.3f})")
                print(f"    文本1: {chunk1['content'][:50]}...")
                print(f"    文本2: {chunk2['content'][:50]}...")
                
                fused_text = self.fuse_texts(
                    chunk1["content"],
                    chunk2["content"]
                )
                
                print(f"    融合后: {fused_text[:50]}...")
                
                # 创建融合后的 chunk
                fused_chunk = {
                    "content": fused_text,
                    "score": max(chunk1.get("score", 0), chunk2.get("score", 0)),
                    "index": chunk1.get("index", idx1),
                    "fused_from": [idx1, idx2],
                    "fusion_method": "t5"
                }
                fused_chunks.append(fused_chunk)
            else:
                # 简单策略：保留分数更高的
                if chunk1.get("score", 0) >= chunk2.get("score", 0):
                    fused_chunk = chunk1.copy()
                    fused_chunk["fused_from"] = [idx1, idx2]
                    fused_chunk["fusion_method"] = "score_based"
                    fused_chunks.append(fused_chunk)
                else:
                    fused_chunk = chunk2.copy()
                    fused_chunk["fused_from"] = [idx1, idx2]
                    fused_chunk["fusion_method"] = "score_based"
                    fused_chunks.append(fused_chunk)
            
            merged_indices.add(idx1)
            merged_indices.add(idx2)
        
        # 添加未融合的 chunks
        for i, chunk in enumerate(chunks):
            if i not in merged_indices:
                fused_chunks.append(chunk)
        
        # 按分数排序
        fused_chunks.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        print(f"  [T5融合] 去重完成，保留 {len(fused_chunks)} 个文本块")
        
        return fused_chunks


class HybridResultCompressor:
    """
    混合结果压缩器
    =============
    结合传统的语义去重和 T5 智能融合
    """
    
    def __init__(
        self, 
        encoder_model,
        t5_fusion_engine: Optional[T5FusionEngine] = None,
        use_t5_fusion: bool = True
    ):
        """
        Args:
            encoder_model: SentenceTransformer 编码器
            t5_fusion_engine: T5 融合引擎（可选）
            use_t5_fusion: 是否使用 T5 融合
        """
        self.encoder = encoder_model
        self.use_t5_fusion = use_t5_fusion and (t5_fusion_engine is not None)
        
        if self.use_t5_fusion:
            self.t5_fusion = t5_fusion_engine
            # 给 T5 引擎也配上 encoder（用于相似度计算）
            self.t5_fusion.encoder = encoder_model
            print("✅ 启用 T5 智能融合模式")
        else:
            self.t5_fusion = None
            print("ℹ️ 使用传统去重模式")
    
    def deduplicate(
        self, 
        chunks: List[Dict], 
        threshold: float = 0.85,
        fusion_threshold: float = 0.75
    ) -> List[Dict]:
        """
        去重（支持 T5 融合）
        
        Args:
            chunks: 检索结果
            threshold: 传统去重阈值（>0.85 直接丢弃）
            fusion_threshold: 融合阈值（0.75-0.85 之间进行融合）
        """
        if len(chunks) <= 1:
            return chunks
        
        if self.use_t5_fusion:
            # 使用 T5 融合去重
            return self.t5_fusion.deduplicate_with_fusion(
                chunks,
                threshold=fusion_threshold,
                enable_fusion=True
            )
        else:
            # 传统贪心去重
            return self._traditional_deduplicate(chunks, threshold)
    
    def _traditional_deduplicate(
        self, 
        chunks: List[Dict], 
        threshold: float = 0.85
    ) -> List[Dict]:
        """传统贪心去重"""
        texts = [c["content"] for c in chunks]
        embs = self.encoder.encode(texts)
        
        keep_indices = [0]
        for i in range(1, len(embs)):
            sims = cosine_similarity([embs[i]], embs[keep_indices])[0]
            if max(sims) < threshold:
                keep_indices.append(i)
        
        return [chunks[i] for i in keep_indices]
    
    def prune_by_length(
        self, 
        chunks: List[Dict], 
        max_total_chars: int = 6000
    ) -> List[Dict]:
        """按总字符数剪枝"""
        if not chunks:
            return chunks
        
        def sort_key(c):
            score = c.get("score", 0)
            length = len(c.get("content", ""))
            return (score, length)
        
        sorted_chunks = sorted(chunks, key=sort_key, reverse=True)
        result = []
        total_chars = 0
        
        for chunk in sorted_chunks:
            chunk_len = len(chunk["content"])
            if total_chars + chunk_len <= max_total_chars:
                result.append(chunk)
                total_chars += chunk_len
            else:
                remaining = max_total_chars - total_chars
                if remaining > 200:
                    truncated = chunk["content"][:remaining]
                    chunk["content"] = truncated
                    result.append(chunk)
                break
        
        return result
    
    def rerank_by_relevance(
        self, 
        query: str, 
        chunks: List[Dict], 
        top_k: int = 5
    ) -> List[Dict]:
        """基于问题-文档相关性重排序"""
        if not chunks:
            return chunks
        
        query_emb = self.encoder.encode([query])[0]
        chunk_texts = [c["content"] for c in chunks]
        chunk_embs = self.encoder.encode(chunk_texts)
        
        similarities = cosine_similarity([query_emb], chunk_embs)[0]
        
        for i, chunk in enumerate(chunks):
            original_score = chunk.get("score", 0)
            rerank_score = float(similarities[i])
            chunk["rerank_score"] = rerank_score
            chunk["combined_score"] = original_score * 0.4 + rerank_score * 0.6
        
        chunks.sort(key=lambda x: x["combined_score"], reverse=True)
        
        return chunks[:top_k]
