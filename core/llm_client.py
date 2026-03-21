import requests
from .utils import count_tokens

class LLMClient:
    """
    通用 LLM 客户端，支持 Ollama 和 OpenAI 兼容 API
    """
    def __init__(self, backend_type="ollama", api_url=None, model_name=None, api_key=None, tokenizer=None):
        self.backend_type = backend_type
        self.api_url = api_url
        self.model_name = model_name
        self.api_key = api_key
        self.tokenizer = tokenizer

    def call(self, prompt, context=None):
        if context:
            context_str = "\n".join(context)
            full_prompt = f"基于以下信息回答问题：\n\n{context_str}\n\n问题：{prompt}\n\n答案："
        else:
            full_prompt = prompt

        prompt_tokens = count_tokens(self.tokenizer, full_prompt)

        if self.backend_type == "ollama":
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": full_prompt}],
                "stream": False
            }
            response = requests.post(self.api_url, json=payload)
            if response.status_code == 200:
                result = response.json()
                answer = result['message']['content']
                answer_tokens = count_tokens(self.tokenizer, answer)
                return answer, prompt_tokens, answer_tokens
            else:
                raise Exception(f"Ollama 调用失败: {response.status_code} - {response.text}")

        elif self.backend_type == "openai":
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": full_prompt}],
                "stream": False
            }
            response = requests.post(self.api_url, json=payload, headers=headers)
            if response.status_code == 200:
                result = response.json()
                answer = result['choices'][0]['message']['content']
                answer_tokens = count_tokens(self.tokenizer, answer)
                return answer, prompt_tokens, answer_tokens
            else:
                raise Exception(f"OpenAI API 调用失败: {response.status_code} - {response.text}")

        else:
            raise ValueError(f"不支持的 backend_type: {self.backend_type}")