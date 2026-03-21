import gradio as gr
import torch
import numpy as np
import os
from transformers import AutoTokenizer
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import normalize

from core.compressor import HardCompressor
from core.merger import PromptMerger
from core.indexer import Indexer
from core.llm_client import LLMClient
from core.summarizer import DialogSummarizer
from core.utils import split_sentences, count_tokens, chinese_word_seg

import config

# ========== 全局变量 ==========
_llm_client = None          # 当前 LLM 客户端实例
_current_config = {}        # 当前配置 (用于判断是否需要重建)

def get_or_create_llm_client(backend_type, api_url, model_name, api_key, tokenizer):
    """根据配置创建或返回 LLM 客户端（全局单例，当配置变化时重建）"""
    global _llm_client, _current_config
    new_config = {
        'backend_type': backend_type,
        'api_url': api_url,
        'model_name': model_name,
        'api_key': api_key
    }
    if _llm_client is None or _current_config != new_config:
        _llm_client = LLMClient(
            backend_type=backend_type,
            api_url=api_url,
            model_name=model_name,
            api_key=api_key,
            tokenizer=tokenizer
        )
        _current_config = new_config
        print(f"LLM 客户端已更新: {backend_type} - {model_name} @ {api_url}")
    return _llm_client

# ========== 加载固定模型 ==========
print("正在加载模型...")
compressor = HardCompressor(config.MODEL_PATH)
tokenizer = compressor.tokenizer
merger = PromptMerger(config.ENCODER_NAME, config.SIMILARITY_THRESHOLD)
summarizer = DialogSummarizer(
    base_model_path=config.SUMMARIZER_BASE,
    adapter_path=config.SUMMARIZER_ADAPTER,
    adapter_name=config.SUMMARIZER_ADAPTER_NAME
)

# 初始化默认 LLM 客户端（Ollama）
_llm_client = LLMClient(
    backend_type="ollama",
    api_url=config.OLLAMA_URL,
    model_name=config.MODEL_NAME,
    tokenizer=tokenizer
)
_current_config = {
    'backend_type': "ollama",
    'api_url': config.OLLAMA_URL,
    'model_name': config.MODEL_NAME,
    'api_key': None
}
print("所有模型加载完成。")

# ========== 多文档处理函数 ==========
def process_multiple_files(files, text_input, progress=gr.Progress()):
    all_fragments = []
    all_vectors = []
    doc_ids = []
    doc_orders = []

    docs_content = []
    doc_name_list = []

    if files is not None:
        for file in files:
            try:
                with open(file.name, 'r', encoding='utf-8') as f:
                    content = f.read()
                docs_content.append(content)
                doc_name_list.append(os.path.basename(file.name))
            except Exception as e:
                return [], np.array([]), [], []
    if text_input:
        docs_content.append(text_input)
        doc_name_list.append("直接输入文本")

    if not docs_content:
        return [], np.array([]), [], []

    progress(0, desc="开始处理多文档...")
    total_docs = len(docs_content)
    for doc_idx, (content, name) in enumerate(zip(docs_content, doc_name_list)):
        progress(0.1 + 0.8 * doc_idx / total_docs, desc=f"处理文档 {doc_idx+1}/{total_docs}: {name}")
        sentences = split_sentences(content)
        if not sentences:
            continue

        compressed = []
        for sent in sentences:
            comp, _, _ = compressor.compress(sent)
            if comp:
                compressed.append(comp)

        if compressed:
            result = merger.process(compressed, do_merge=True)
            merged_frags = result['fragments']
            merged_vecs = result['vectors']
            for j, (frag, vec) in enumerate(zip(merged_frags, merged_vecs)):
                all_fragments.append(frag)
                all_vectors.append(vec)
                doc_ids.append(doc_idx)
                doc_orders.append(j)

    if not all_fragments:
        return [], np.array([]), [], []

    all_vectors = np.array(all_vectors)
    progress(1.0, desc="完成")
    return all_fragments, all_vectors, doc_ids, doc_orders

# ========== 混合检索函数 ==========
def hybrid_search(question, fragments, vectors, doc_ids, doc_orders, alpha=0.5, top_k=3, bm25_candidates=30):
    query_vec = merger.encoder.encode([question])
    dim = vectors.shape[1]
    indexer = Indexer(dim)
    indexer.build(vectors, fragments)
    dense_scores, dense_indices = indexer.search(query_vec, bm25_candidates)
    dense_score_dict = {idx: score for idx, score in zip(dense_indices, dense_scores)}

    tokenized_fragments = [chinese_word_seg(frag) for frag in fragments]
    bm25 = BM25Okapi(tokenized_fragments)
    query_tokens = chinese_word_seg(question)
    bm25_scores = bm25.get_scores(query_tokens)
    bm25_score_dict = {i: bm25_scores[i] for i in range(len(fragments))}

    candidate_indices = set(dense_indices)
    combined = []
    for idx in candidate_indices:
        dense = dense_score_dict[idx]
        bm25_val = bm25_score_dict[idx]
        dense_norm = (dense + 1) / 2
        bm25_vals = [bm25_score_dict[i] for i in candidate_indices]
        bm25_min, bm25_max = min(bm25_vals), max(bm25_vals)
        if bm25_max > bm25_min:
            bm25_norm = (bm25_val - bm25_min) / (bm25_max - bm25_min)
        else:
            bm25_norm = 0.5
        combined_score = alpha * dense_norm + (1 - alpha) * bm25_norm
        combined.append((idx, combined_score))
    combined.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in combined[:top_k]]

# ========== 问答函数（使用全局 LLM 客户端） ==========
def answer_question(question, fragments, vectors, doc_ids, doc_orders, history_summary, retriever_mode):
    global _llm_client

    has_docs = fragments is not None and len(fragments) > 0

    # 将滑块值映射为混合检索的 alpha
    alpha = (retriever_mode + 1) / 2
    alpha = max(0.0, min(1.0, alpha))

    # 编码问题向量
    query_vec = merger.encoder.encode([question])
    query_vec_norm = query_vec / np.linalg.norm(query_vec)

    # 文档相关性
    max_doc_sim = 0.0
    if has_docs:
        normalized_vectors = normalize(vectors, norm='l2')
        similarities = np.dot(normalized_vectors, query_vec_norm.T).flatten()
        max_doc_sim = float(np.max(similarities))

    DOC_RELEVANCE_THRESHOLD = getattr(config, 'DOC_RELEVANCE_THRESHOLD', 0.5)

    # 摘要相关性
    use_summary = False
    summary_sim = 0.0
    if history_summary:
        summary_vec = merger.encoder.encode([history_summary])
        summary_vec_norm = summary_vec / np.linalg.norm(summary_vec)
        summary_sim = float((query_vec_norm @ summary_vec_norm.T).item())
        SUMMARY_RELEVANCE_THRESHOLD = getattr(config, 'SUMMARY_RELEVANCE_THRESHOLD', 0.4)
        use_summary = summary_sim >= SUMMARY_RELEVANCE_THRESHOLD

    context = []
    if use_summary:
        context.append(f"对话历史摘要：{history_summary}")

    expanded_fragments = []
    if has_docs and max_doc_sim >= DOC_RELEVANCE_THRESHOLD:
        final_indices = hybrid_search(
            question, fragments, vectors, doc_ids, doc_orders,
            alpha=alpha,
            top_k=config.TOP_K,
            bm25_candidates=config.BM25_TOP_K
        )

        all_indices = set()
        for idx in final_indices:
            doc_id = doc_ids[idx]
            order = doc_orders[idx]
            all_indices.add(idx)
            if order > 0:
                for j, (d, o) in enumerate(zip(doc_ids, doc_orders)):
                    if d == doc_id and o == order - 1:
                        all_indices.add(j)
                        break
            max_order_in_doc = max(o for d, o in zip(doc_ids, doc_orders) if d == doc_id)
            if order < max_order_in_doc:
                for j, (d, o) in enumerate(zip(doc_ids, doc_orders)):
                    if d == doc_id and o == order + 1:
                        all_indices.add(j)
                        break

        sorted_indices = sorted(all_indices)
        expanded_fragments = [fragments[i] for i in sorted_indices]
        context.extend(expanded_fragments)

    try:
        answer, prompt_tokens, answer_tokens = _llm_client.call(question, context=context)
        retrieved_tokens = sum([count_tokens(tokenizer, f) for f in expanded_fragments]) if expanded_fragments else 0

        if alpha == 1.0:
            mode_desc = "语义检索 (alpha=1)"
        elif alpha == 0.0:
            mode_desc = "关键词检索 (alpha=0)"
        else:
            mode_desc = f"混合检索 (alpha={alpha:.2f})"

        token_stats = (
            f"输入 prompt Token 数: {prompt_tokens}\n"
            f"输出答案 Token 数: {answer_tokens}\n"
            f"检索片段总 Token 数: {retrieved_tokens}\n"
            f"检索模式: {mode_desc}\n"
        )
        if has_docs:
            token_stats += f"文档最大相关性: {max_doc_sim:.3f}\n"
        if use_summary:
            token_stats += f"摘要相关性: {summary_sim:.3f} (已加入)\n"
        else:
            token_stats += "摘要相关性不足，未加入\n"
        if not has_docs:
            token_stats += "未提供文档，仅基于对话摘要回答\n"

        # 生成新摘要
        current_dialog = f"Human: {question}\nAssistant: {answer}"
        if history_summary:
            full_dialog = f"历史摘要：{history_summary}\n{current_dialog}"
        else:
            full_dialog = current_dialog
        new_summary = summarizer.summarize(full_dialog, max_length=config.MAX_SUMMARY_TOKENS)

        return answer, token_stats, new_summary
    except Exception as e:
        return f"错误: {e}", "", history_summary

# ========== 更新模型配置的回调 ==========
def update_model(backend_type, api_url, model_name, api_key):
    """根据用户输入重新创建 LLM 客户端并返回状态消息"""
    try:
        global _llm_client, _current_config
        api_url = api_url.strip()
        model_name = model_name.strip()
        api_key = api_key.strip() if api_key else None

        if not api_url or not model_name:
            return "请填写完整的 API 地址和模型名称"

        # 针对 openai 模式，自动补全路径
        if backend_type == "openai":
            # 自动将模型名转为小写（阿里云 DashScope 等要求）
            model_name = model_name.lower()
            if not api_url.endswith("/chat/completions"):
                if api_url.endswith("/"):
                    api_url = api_url.rstrip("/")
                api_url = api_url + "/chat/completions"

        # 创建新客户端
        _llm_client = LLMClient(
            backend_type=backend_type,
            api_url=api_url,
            model_name=model_name,
            api_key=api_key,
            tokenizer=tokenizer
        )
        _current_config = {
            'backend_type': backend_type,
            'api_url': api_url,
            'model_name': model_name,
            'api_key': api_key
        }
        return f"模型配置已更新！当前使用 {backend_type} - {model_name}"
    except Exception as e:
        return f"更新失败: {str(e)}"


# ========== Gradio 界面 ==========
with gr.Blocks(title="Prompt Composer (可调检索模式+多后端)") as demo:
    gr.Markdown("# Prompt Composer 工具（多文档检索 + 每轮对话摘要）")
    gr.Markdown("上传多个文本文件或直接输入文本（可选），系统将压缩、合并并建立索引。如不上传文档，系统将仅基于对话历史回答。")

    # 折叠的模型配置区域
    with gr.Accordion("模型配置（切换后端）", open=False):
        with gr.Row():
            backend_radio = gr.Radio(
                choices=["ollama", "openai"],
                value="ollama",
                label="后端类型"
            )
            api_url_input = gr.Textbox(
                label="API 地址",
                value=config.OLLAMA_URL if config.OLLAMA_URL else "http://localhost:11434/api/chat",
                placeholder="Ollama: http://localhost:11434/api/chat\nOpenAI 兼容: https://dashscope.aliyuncs.com/compatible-mode/v1\n(代码会自动补全 /chat/completions)"
            )
            model_name_input = gr.Textbox(
                label="模型名称",
                value=config.MODEL_NAME if config.MODEL_NAME else "qwen2.5:3b",
                placeholder="Ollama: qwen2.5:3b\nOpenAI 兼容: qwen-turbo, qwen-plus, qwen-max 等"
            )
            api_key_input = gr.Textbox(
                label="API Key（可选）",
                type="password",
                placeholder="仅 OpenAI 兼容模式需要（如阿里云 DashScope API Key）"
            )
        update_btn = gr.Button("更新模型配置", variant="secondary")
        config_status = gr.Textbox(label="配置状态", interactive=False)

    # 提问与答案区域
    with gr.Row():
        with gr.Column(scale=1):
            question_input = gr.Textbox(label="请输入您的问题", lines=8)
            with gr.Row():
                ask_btn = gr.Button("提问", variant="secondary")
                retriever_slider = gr.Slider(
                    minimum=-1.0,
                    maximum=1.0,
                    step=0.1,
                    value=0.0,
                    label="检索模式",
                    info="-1: 关键词检索  0: 平衡混合  1: 语义检索"
                )
        with gr.Column(scale=1):
            answer_output = gr.Textbox(label="答案", lines=8, interactive=False)

    # 文档处理区域
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="上传多个文本文件", file_count="multiple", file_types=[".txt"])
            text_input = gr.Textbox(label="或直接输入文本（作为额外文档）", lines=5)
            process_btn = gr.Button("处理文档", variant="primary")
        with gr.Column(scale=1):
            history_summary_output = gr.Textbox(
                label="当前对话摘要",
                lines=15,
                max_lines=20,
                interactive=False,
                placeholder="暂无对话历史",
                autoscroll=False
            )

    fragments_state = gr.State()
    vectors_state = gr.State()
    doc_ids_state = gr.State()
    doc_orders_state = gr.State()
    history_summary_state = gr.State("")

    def reset_on_process(files, text_input):
        fragments, vectors, doc_ids, doc_orders = process_multiple_files(files, text_input)
        return fragments, vectors, doc_ids, doc_orders, ""

    process_btn.click(
        fn=reset_on_process,
        inputs=[file_input, text_input],
        outputs=[fragments_state, vectors_state, doc_ids_state, doc_orders_state, history_summary_state]
    )

    gr.Markdown("---")
    gr.Markdown("## Token 统计")
    token_stats_output = gr.Textbox(label="Token 统计", lines=8, interactive=False)

    ask_btn.click(
        fn=answer_question,
        inputs=[question_input, fragments_state, vectors_state, doc_ids_state, doc_orders_state, history_summary_state, retriever_slider],
        outputs=[answer_output, token_stats_output, history_summary_state]
    ).then(
        fn=lambda summary: summary,
        inputs=history_summary_state,
        outputs=history_summary_output
    )

    update_btn.click(
        fn=update_model,
        inputs=[backend_radio, api_url_input, model_name_input, api_key_input],
        outputs=config_status
    )

    gr.Markdown("注意：每轮对话后自动生成摘要，作为下一轮的历史上下文。")

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)