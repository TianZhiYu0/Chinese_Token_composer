import jieba
import jieba.posseg as pseg
from typing import List


class QueryExpander:
    """查询扩展器：同义词替换 + 实体识别，无 LLM 依赖"""

    # 内置简单同义词映射（可扩展或从文件加载）
    SYNONYMS = {
        "营收": ["收入", "销售额", "营业收入"],
        "利润": ["净利润", "盈利", "收益"],
        "增长": ["增加", "上升", "提高", "同比增长"],
        "减少": ["下降", "降低", "下滑"],
        "成本": ["费用", "支出", "开销"],
        "毛利率": ["毛利润", "销售毛利率"],
        "净利率": ["净利润率", "销售净利率"],
        "同比": ["较去年同期", "与去年相比"],
        "环比": ["较上期", "与上期相比"],
    }

    def __init__(self, use_ner: bool = True):
        self.use_ner = use_ner

    def expand(self, query: str, max_expansions: int = 3) -> List[str]:
        """
        返回扩展后的查询列表（包含原查询）
        """
        expansions = [query]
        words = list(jieba.cut(query))

        # 同义词替换
        for word in words:
            if word in self.SYNONYMS:
                for syn in self.SYNONYMS[word][:2]:  # 每词最多扩展2个
                    new_q = query.replace(word, syn)
                    if new_q not in expansions:
                        expansions.append(new_q)

        # 实体识别增强（提取关键实体，可构造更精确查询）
        if self.use_ner:
            entities = []
            for pair in pseg.cut(query):
                word, flag = pair.word, pair.flag
                if flag in ['nr', 'ns', 'nt', 'nz']:  # 人名、地名、机构、专名
                    entities.append(word)
            # 若实体较多，可生成组合查询（略）

        return expansions[:max_expansions]

    def extract_key_entities(self, query: str) -> List[str]:
        """提取关键实体，用于加权检索"""
        entities = []
        for pair in pseg.cut(query):
            word, flag = pair.word, pair.flag
            if flag in ['nr', 'ns', 'nt', 'nz']:
                entities.append(word)
        return entities