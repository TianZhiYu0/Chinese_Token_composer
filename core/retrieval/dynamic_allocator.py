"""
动态Token分配器
================

根据问题复杂度和文档长度动态调整检索文档和历史摘要的比例

优化策略：
1. 简单问题 → 减少文档tokens（500-800）
2. 复杂问题 → 增加文档tokens（1000-1500）
3. 历史摘要 → 固定200-300 tokens
"""
import re


class DynamicTokenAllocator:
    """动态Token分配器"""
    
    def __init__(self, max_total_tokens=2500):
        """
        Args:
            max_total_tokens: 最大总token数
        """
        self.max_total_tokens = max_total_tokens
        
    def analyze_question_complexity(self, question: str) -> dict:
        """
        分析问题复杂度
        
        Returns:
            {
                "complexity": "simple" | "medium" | "complex",
                "score": 0.0-1.0,
                "factors": {...}
            }
        """
        factors = {
            "length": len(question),
            "has_numbers": bool(re.search(r'\d+', question)),
            "has_multiple_parts": bool(re.search(r'[，,、;；]', question)),
            "has_analysis_keywords": any(kw in question for kw in [
                "分析", "比较", "为什么", "如何", "评价", "概括", "总结"
            ]),
            "has_comparison": any(kw in question for kw in ["对比", "比较", "异同", "区别"]),
        }
        
        # 计算复杂度分数 (0-1)
        score = 0.0
        score += min(factors["length"] / 100, 0.3)  # 长度贡献最多30%
        score += 0.1 if factors["has_numbers"] else 0  # 数字问题
        score += 0.15 if factors["has_multiple_parts"] else 0  # 多部分问题
        score += 0.25 if factors["has_analysis_keywords"] else 0  # 分析类问题
        score += 0.2 if factors["has_comparison"] else 0  # 比较类问题
        
        # 确定复杂度等级
        if score < 0.4:
            complexity = "simple"
        elif score < 0.7:
            complexity = "medium"
        else:
            complexity = "complex"
        
        return {
            "complexity": complexity,
            "score": score,
            "factors": factors
        }
    
    def allocate_tokens(self, question: str, available_chunks_tokens: int) -> dict:
        """
        动态分配token
        
        Args:
            question: 用户问题
            available_chunks_tokens: 可用文档总tokens
            
        Returns:
            {
                "retrieval_tokens": int,  # 分配给检索文档的tokens
                "history_tokens": int,    # 分配给历史摘要的tokens
                "ratio": float            # 检索占比
            }
        """
        # 分析复杂度
        analysis = self.analyze_question_complexity(question)
        complexity = analysis["complexity"]
        
        # 基础分配策略
        if complexity == "simple":
            # 简单问题：文档500-800 tokens，历史200 tokens
            retrieval_budget = min(800, available_chunks_tokens)
            history_budget = 200
        elif complexity == "medium":
            # 中等问题：文档800-1200 tokens，历史300 tokens
            retrieval_budget = min(1200, available_chunks_tokens)
            history_budget = 300
        else:
            # 复杂问题：文档1200-1500 tokens，历史400 tokens
            retrieval_budget = min(1500, available_chunks_tokens)
            history_budget = 400
        
        # 确保总token不超过上限
        total_needed = retrieval_budget + history_budget
        if total_needed > self.max_total_tokens:
            # 按比例缩减
            scale = self.max_total_tokens / total_needed
            retrieval_budget = int(retrieval_budget * scale)
            history_budget = int(history_budget * scale)
        
        ratio = retrieval_budget / (retrieval_budget + history_budget)
        
        return {
            "retrieval_tokens": retrieval_budget,
            "history_tokens": history_budget,
            "ratio": ratio,
            "complexity": complexity,
            "complexity_score": analysis["score"]
        }


class SmartDocumentExtractor:
    """智能文档信息提取器"""
    
    @staticmethod
    def extract_key_info(text: str, max_chars: int = 500) -> str:
        """
        提取文档关键信息
        
        策略：
        1. 保留首句（通常包含核心信息）
        2. 提取包含关键实体的句子（数字、时间、人名）
        3. 删除冗余描述和过渡语句
        
        Args:
            text: 原始文档
            max_chars: 最大字符数
            
        Returns:
            精简后的文档
        """
        if len(text) <= max_chars:
            # 即使文本较短，也进行压缩优化
            return SmartDocumentExtractor._compress_short_text(text, max_chars)
        
        sentences = re.split(r'[。；;！!？?]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return text[:max_chars]
        
        # 策略：智能选择关键句
        result_parts = []
        
        # 1. 保留首句（核心信息）
        if sentences:
            result_parts.append(sentences[0])
        
        # 2. 提取包含关键信息的句子
        key_sentences = []
        for sent in sentences[1:-1]:
            # 检查是否包含关键信息
            has_key_info = (
                re.search(r'\d{4}年', sent) or  # 年份
                re.search(r'\d+%', sent) or      # 百分比
                re.search(r'\d+[万千百]元', sent) or   # 金额
                re.search(r'\d+[个项人]', sent) or    # 数量
                re.search(r'[\u4e00-\u9fa5]{2,4}(同志|主席|部长|书记|省长)', sent)  # 人名+职位
            )
            if has_key_info:
                key_sentences.append(sent)
        
        # 3. 添加关键句（最多2句）
        result_parts.extend(key_sentences[:2])
        
        # 4. 添加尾句（结论/结果）
        if len(sentences) > 1:
            result_parts.append(sentences[-1])
        
        result = '。'.join(result_parts) + '。'
        
        # 如果仍然超长，截断
        if len(result) > max_chars:
            result = result[:max_chars] + '...'
        
        return result
    
    @staticmethod
    def _compress_short_text(text: str, max_chars: int) -> str:
        """压缩较短文本，去除冗余描述"""
        if len(text) <= max_chars * 0.7:
            return text
        
        sentences = re.split(r'[。；;！!？?]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= 2:
            return text[:max_chars]
        
        # 保留关键句
        key_parts = []
        for sent in sentences:
            # 优先保留包含数字/时间的句子
            if re.search(r'\d', sent) or len(sent) > 20:
                key_parts.append(sent)
                if len('。'.join(key_parts)) > max_chars * 0.8:
                    break
        
        if key_parts:
            return '。'.join(key_parts) + '。'
        
        return text[:max_chars]
    
    @staticmethod
    def extract_by_importance(chunks: list, max_total_chars: int = 2000) -> list:
        """
        按重要性提取文档
        
        Args:
            chunks: 文档列表，每个包含 content 和 score
            max_total_chars: 最大总字符数
            
        Returns:
            精简后的文档列表
        """
        if not chunks:
            return chunks
        
        # 按分数排序
        sorted_chunks = sorted(chunks, key=lambda x: x.get("score", 0), reverse=True)
        
        result = []
        total_chars = 0
        
        for chunk in sorted_chunks:
            content = chunk["content"]
            
            # 计算该chunk分配的字符数
            remaining = max_total_chars - total_chars
            if remaining <= 0:
                break
            
            # 高分数chunk保留更多，低分数chunk精简
            score = chunk.get("score", 0)
            if score > 0.8:
                # 高相关度：保留80%
                chunk_budget = int(remaining * 0.8)
            elif score > 0.6:
                # 中相关度：保留50%
                chunk_budget = int(remaining * 0.5)
            else:
                # 低相关度：保留30%
                chunk_budget = int(remaining * 0.3)
            
            # 提取关键信息
            if len(content) > chunk_budget:
                extracted = SmartDocumentExtractor.extract_key_info(content, chunk_budget)
            else:
                extracted = content
            
            # 创建新chunk
            new_chunk = chunk.copy()
            new_chunk["content"] = extracted
            new_chunk["original_length"] = len(content)
            new_chunk["extracted_length"] = len(extracted)
            result.append(new_chunk)
            
            total_chars += len(extracted)
        
        return result
