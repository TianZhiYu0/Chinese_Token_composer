import re
import jieba

def count_tokens(tokenizer, text):
    """使用给定的 tokenizer 计算 token 数"""
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