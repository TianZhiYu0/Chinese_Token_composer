import requests
from .utils import count_tokens

class OllamaClient:
    def __init__(self, base_url, model_name, tokenizer):
        self.base_url = base_url
        self.model_name = model_name
        self.tokenizer = tokenizer   # 用于 token 计数

    def call(self, prompt, context=None):
        """
        调用 Ollama 生成答案
        context: 上下文片段列表（如检索到的片段）
        返回 (answer, prompt_tokens, answer_tokens)
        """
        if context:
            context_str = "\n".join(context)
            full_prompt = f"基于以下信息回答问题：\n\n{context_str}\n\n问题：{prompt}\n\n答案："
        else:
            full_prompt = prompt

        prompt_tokens = count_tokens(self.tokenizer, full_prompt)

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": full_prompt}],
            "stream": False
        }
        response = requests.post(self.base_url, json=payload)
        if response.status_code == 200:
            result = response.json()
            answer = result['message']['content']
            answer_tokens = count_tokens(self.tokenizer, answer)
            return answer, prompt_tokens, answer_tokens
        else:
            raise Exception(f"Ollama 调用失败: {response.status_code} - {response.text}")