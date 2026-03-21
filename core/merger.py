import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class PromptMerger:
    def __init__(self, encoder_model_name, similarity_threshold=0.8):
        self.encoder = SentenceTransformer(encoder_model_name)
        self.similarity_threshold = similarity_threshold

    def encode_fragments(self, fragments):
        return self.encoder.encode(fragments, convert_to_tensor=False)

    def merge_consecutive_fragments(self, fragments, vectors, similarity_threshold=None):
        if similarity_threshold is None:
            similarity_threshold = self.similarity_threshold
        merged_fragments = []
        merged_vectors = []
        merge_map = [-1] * len(fragments)
        i = 0
        while i < len(fragments):
            current = fragments[i]
            current_vec = vectors[i]
            j = i + 1
            while j < len(fragments):
                sim = cosine_similarity([current_vec], [vectors[j]])[0][0]
                if sim >= similarity_threshold:
                    current += " " + fragments[j]
                    current_vec = (current_vec + vectors[j]) / 2
                    j += 1
                else:
                    break
            merged_fragments.append(current)
            merged_vectors.append(current_vec)
            for idx in range(i, j):
                merge_map[idx] = len(merged_fragments) - 1
            i = j
        return merged_fragments, np.array(merged_vectors), merge_map

    def process(self, fragments, do_merge=True):
        vectors = self.encode_fragments(fragments)
        if do_merge:
            final_fragments, final_vectors, merge_map = self.merge_consecutive_fragments(fragments, vectors)
        else:
            final_fragments, final_vectors, merge_map = fragments, vectors, list(range(len(fragments)))
        return {
            'fragments': final_fragments,
            'vectors': final_vectors,
            'merge_map': merge_map
        }