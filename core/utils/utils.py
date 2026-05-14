import re
import jieba

def count_tokens(tokenizer, text):
    """使用给定的 tokenizer 计算 token 数"""
    if tokenizer is None:
        # 如果没有 tokenizer，使用简单的字符数估算
        # 中文：1字符 ≈ 1.5 tokens，英文：1单词 ≈ 1.3 tokens
        if text is None:
            return 0
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        english_words = len(text.split())
        return int(chinese_chars * 1.5 + english_words * 1.3)
    
    if text is None:
        return 0
    
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)

def split_sentences(text):
    """按中文标点及换行符分割句子"""
    sentences = re.split(r'[。！？\n]', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def chinese_word_seg(text):
    """中文分词，使用 jieba（如果已安装），否则用空格简单切分"""
    try:
        import jieba
        return list(jieba.cut(text))
    except ImportError:
        return text.split()