import torch  # 添加这一行
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class PromptMerger:
    def __init__(self, encoder_name, similarity_threshold=0.8):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 确保使用本地模型，不从 HuggingFace 下载
        import os
        if not os.path.exists(encoder_name):
            raise FileNotFoundError(f"模型路径不存在: {encoder_name}")
        
        print(f"正在加载句子编码器: {encoder_name}")
        print(f"设备: {device}")
        
        # 使用 local_files_only=True 强制使用本地模型
        self.encoder = SentenceTransformer(
            encoder_name, 
            device=device,
            local_files_only=True  # 关键：禁止联网下载
        )
        
        self.similarity_threshold = similarity_threshold
        print("句子编码器加载成功")

    def process(self, fragments, do_merge=True):
        # 原有逻辑保持不变
        if not fragments:
            return {'fragments': [], 'vectors': []}
        embeddings = self.encoder.encode(fragments, convert_to_tensor=False)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        if not do_merge:
            return {'fragments': fragments, 'vectors': embeddings}
        # 合并相似片段
        merged_frags = []
        merged_vecs = []
        i = 0
        while i < len(fragments):
            current_frag = fragments[i]
            current_vec = embeddings[i]
            j = i + 1
            while j < len(fragments):
                sim = cosine_similarity([current_vec], [embeddings[j]])[0][0]
                if sim >= self.similarity_threshold:
                    current_frag += " " + fragments[j]
                    current_vec = (current_vec + embeddings[j]) / 2
                    j += 1
                else:
                    break
            merged_frags.append(current_frag)
            merged_vecs.append(current_vec)
            i = j
        return {'fragments': merged_frags, 'vectors': np.array(merged_vecs)}