from typing import List, Dict, Any
import re
from collections import defaultdict

class ContextBuilder:
    """上下文构建器：信息密度加权 + Token 预算分配 + 句子级截断"""
    
    def __init__(self, tokenizer, max_tokens: int = 2000):
        """
        Args:
            tokenizer: Token 计数器
            max_tokens: 最大 Token 预算（优化：4000→2000）
        """
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        
        # 优化配置
        self.token_elasticity = 0.1  # 10% 弹性
        self.min_sentence_tokens = 8  # 最小保留句子长度
        self.density_weight = 0.3  # 信息密度权重
        self.max_chunks_per_doc = 2  # 每篇文档最多 2 个片段
        self.chunk_similarity_threshold = 0.7  # 重复过滤阈值
    
    def count_tokens(self, text: str) -> int:
        """粗略估算 token 数（中文约 0.5 token/字）"""
        return len(text) // 2
    
    def _calc_density(self, text: str) -> float:
        """
        计算信息密度
        density = (关键实体数 + 数字数) / 片段长度
        """
        if not text:
            return 0.0
        
        # 统计关键信息
        key_entities = 0
        key_entities += len(re.findall(r'\d{4}年', text))  # 年份
        key_entities += len(re.findall(r'\d+%', text))  # 百分比
        key_entities += len(re.findall(r'\d+[万千百]元', text))  # 金额
        key_entities += len(re.findall(r'\d+[个项人]', text))  # 数量
        key_entities += len(re.findall(r'[\u4e00-\u9fa5]{2,4}(同志|主席|部长|书记|省长)', text))  # 人名+职位
        
        # 密度 = 关键信息数 / 长度
        density = key_entities / max(len(text) / 100, 1)  # 归一化到每100字
        return density
    
    def _truncate_by_sentence(self, text: str, target_tokens: int) -> str:
        """
        句子级截断（严禁在句子中间切断）
        
        Args:
            text: 原始文本
            target_tokens: 目标 token 数
            
        Returns:
            截断后的文本（保留完整句子）
        """
        if self.count_tokens(text) <= target_tokens:
            return text
        
        # 按句子分割
        sentences = re.split(r'([。；;！!？?])', text)
        sentences = [s for s in sentences if s.strip()]
        
        # 贪心选取句子
        result = []
        total_tokens = 0
        
        for i in range(0, len(sentences), 2):  # 跳过分隔符
            sent = sentences[i] if i < len(sentences) else ""
            punct = sentences[i+1] if i+1 < len(sentences) else ""
            
            sent_tokens = self.count_tokens(sent + punct)
            if total_tokens + sent_tokens <= target_tokens:
                result.append(sent + punct)
                total_tokens += sent_tokens
            else:
                # 检查是否超过最小保留长度
                if total_tokens > self.min_sentence_tokens:
                    break
        
        return ''.join(result)
    
    def _filter_similar_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        过滤重复片段（基于 Rouge-L 简化版）
        
        Args:
            chunks: 片段列表
            
        Returns:
            去重后的片段列表
        """
        if len(chunks) <= 1:
            return chunks
        
        selected = [chunks[0]]
        
        for chunk in chunks[1:]:
            is_duplicate = False
            new_content = chunk["content"]
            
            for sel in selected:
                sel_content = sel["content"]
                
                # 简化的 Rouge-L：计算最长公共子序列比例
                if len(new_content) < 20 or len(sel_content) < 20:
                    continue
                
                # 简化的相似度计算（基于字符重叠）
                new_chars = set(new_content[:200])  # 只比较前200字
                sel_chars = set(sel_content[:200])
                
                if len(new_chars) > 0 and len(sel_chars) > 0:
                    overlap = len(new_chars & sel_chars) / min(len(new_chars), len(sel_chars))
                    
                    if overlap > self.chunk_similarity_threshold:
                        # 检查是否包含额外关键信息
                        new_numbers = len(re.findall(r'\d+', new_content))
                        sel_numbers = len(re.findall(r'\d+', sel_content))
                        
                        if new_numbers <= sel_numbers:
                            is_duplicate = True
                            break
            
            if not is_duplicate:
                selected.append(chunk)
        
        return selected
    
    def build(self, retrieved_chunks: List[Dict], history_context: str,
              retrieval_ratio: float = 0.7) -> str:
        """
        构建优化后的上下文字符串
        
        优化策略：
        1. 信息密度加权排序
        2. 句子级截断
        3. 重复内容过滤
        4. 同文档限制
        
        Args:
            retrieved_chunks: 检索到的片段列表，每个含 content, score
            history_context: 历史上下文字符串
            retrieval_ratio: 检索内容所占 token 比例
        """
        retrieval_budget = int(self.max_tokens * retrieval_ratio)
        history_budget = self.max_tokens - retrieval_budget
        
        # ========== 1. 信息密度加权排序 ==========
        for chunk in retrieved_chunks:
            if "density" not in chunk:
                chunk["density"] = self._calc_density(chunk["content"])
            # 有效分数 = 相关性 * (1 + 密度权重 * 信息密度)
            chunk["effective_score"] = chunk.get("score", 0) * (1 + self.density_weight * chunk["density"])
        
        # 按有效分数降序排序
        sorted_chunks = sorted(retrieved_chunks, key=lambda x: x.get("effective_score", 0), reverse=True)
        
        # ========== 2. 重复过滤 ==========
        filtered_chunks = self._filter_similar_chunks(sorted_chunks)
        
        # ========== 3. 贪心选取 + 同文档限制 ==========
        selected = []
        doc_count = defaultdict(int)  # 记录每篇文档的片段数
        total_tokens = 0
        
        # 弹性预算上限
        max_budget = int(retrieval_budget * (1 + self.token_elasticity))
        
        for chunk in filtered_chunks:
            doc_id = chunk.get("doc_id", "unknown")
            
            # 同文档限制
            if doc_count[doc_id] >= self.max_chunks_per_doc:
                continue
            
            chunk_tokens = self.count_tokens(chunk["content"])
            
            # 检查是否超过预算
            if total_tokens + chunk_tokens <= max_budget:
                # 直接添加
                selected.append(chunk)
                total_tokens += chunk_tokens
                doc_count[doc_id] += 1
            else:
                # 尝试句子级截断
                remaining = retrieval_budget - total_tokens
                if remaining > self.min_sentence_tokens:
                    truncated = self._truncate_by_sentence(chunk["content"], remaining)
                    if truncated and len(truncated) > 20:  # 确保截断后仍有内容
                        chunk["content"] = truncated
                        selected.append(chunk)
                        total_tokens += self.count_tokens(truncated)
                        doc_count[doc_id] += 1
                break
        
        # ========== 4. 拼接检索内容 ==========
        retrieval_text = "\n\n".join([c["content"] for c in selected])
        
        # ========== 5. 历史上下文截断 ==========
        history_text = history_context
        if self.count_tokens(history_text) > history_budget:
            # 保留尾部（最新）内容，句子级截断
            history_text = self._truncate_by_sentence(history_text, history_budget)
        
        return f"【相关文档】\n{retrieval_text}\n\n【对话历史】\n{history_text}"