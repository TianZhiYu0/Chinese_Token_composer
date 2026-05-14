import gradio as gr
import numpy as np
import os
import torch
from collections import defaultdict

# -------------------- 核心模块导入 --------------------
from core.compression import HardCompressor, PromptMerger, DocumentPreprocessor, KeywordRetriever, HybridCompressor, WordPriorityCompressor
from core.engine import QAEngine, DialogSummarizer
from core.model import LLMClient
from core.visualization import visualize_tree
from core.io import DocumentReader
from core.utils import split_sentences, count_tokens
import config

from core.compression.compression_strategy import CompressionStrategy

# ==================== 全局初始化 ====================
_llm_client = None
_current_config = {}
_qa_engine = None
_current_compressor_model = "mooscomp"
_compressor = None
_compressor_type = "hard"  # 跟踪压缩器类型: "hard", "word_priority", "hybrid"

_strategy = None
_keyword_retriever = None
_documents_cache = None


def switch_compressor_model(model_type, device=None):
    global _compressor, _current_compressor_model, tokenizer, _compressor_type
    if device is None:
        device = config.DEVICE
    
    print(f"\n🔄 正在切换压缩模型: {_current_compressor_model} -> {model_type}")
    
    if model_type in ["mooscomp", "llmlingua2"]:
        _compressor = HardCompressor(model_type=model_type, device=device)
        tokenizer = _compressor.tokenizer
        _compressor_type = "hard"
    elif model_type == "word_priority":
        _compressor = WordPriorityCompressor(
            bert_model_path=config.WORD_PRIORITY_BERT_PATH,
            priority_model_path=config.WORD_PRIORITY_MODEL_PATH,
            device=device
        )
        tokenizer = _compressor.tokenizer
        _compressor_type = "word_priority"
    elif model_type == "hybrid":
        _compressor = HybridCompressor(
            bert_model_path=config.MODEL_PATH,
            word_priority_model_path=config.HYBRID_WORD_PRIORITY_MODEL_PATH,
            mooscomp_model_path=config.MODEL_PATH,
            device=device,
            dynamic_alpha=True,
            fixed_alpha=0.85
        )
        tokenizer = _compressor.tokenizer
        _compressor_type = "hybrid"
    else:
        raise ValueError(f"不支持的压缩模型类型: {model_type}")
    
    _current_compressor_model = model_type
    print(f"✅ 压缩模型已切换为: {model_type} ({_compressor_type})\n")
    return _compressor


_compressor = HardCompressor(model_type="mooscomp", device=config.DEVICE)
tokenizer = _compressor.tokenizer


def get_or_create_llm_client(backend_type, api_url, model_name, api_key, tokenizer):
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


print("正在加载模型...")
compressor = HardCompressor(model_type="mooscomp", device=config.DEVICE)
tokenizer = compressor.tokenizer
merger = PromptMerger(config.ENCODER_NAME, config.SIMILARITY_THRESHOLD)

try:
    summarizer = DialogSummarizer(
        base_model_path=config.SUMMARIZER_BASE,
        adapter_path=config.SUMMARIZER_ADAPTER,
        adapter_name=config.SUMMARIZER_ADAPTER_NAME,
        use_textrank_fallback=True,
        textrank_encoder=merger.encoder
    )
except TypeError:
    summarizer = DialogSummarizer(
        base_model_path=config.SUMMARIZER_BASE,
        adapter_path=config.SUMMARIZER_ADAPTER,
        adapter_name=config.SUMMARIZER_ADAPTER_NAME
    )

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

vector_db_path = os.path.join(config.PROJECT_ROOT, "data", "vector_store.pkl")
_qa_engine = QAEngine(
    llm_client=_llm_client,
    tokenizer=tokenizer,
    encoder=merger.encoder,
    summarizer=summarizer,
    config=config,
    vector_store_path=vector_db_path,
    use_t5_fusion=True,
    use_dynamic_allocation=True
)
print("所有模型加载完成。")


# ==================== 核心处理逻辑 ====================
def process_documents_advanced(
        files, text_input,
        compression_ratio=0.7,
        preprocessing_mode="auto",
        use_keyword_retrieval=False,
        target_tokens=None,
        progress=None
):
    global _strategy, _keyword_retriever, _documents_cache

    if progress is None:
        progress = gr.Progress()

    doc_reader = DocumentReader()
    docs_dict = {}
    if files is not None:
        for file in files:
            try:
                content = doc_reader.read_file(file.name)
                if content and content.strip():
                    docs_dict[os.path.basename(file.name)] = content
            except Exception as e:
                print(f"处理文件出错 {file.name}: {e}")
                continue
    if text_input:
        docs_dict["直接输入文本"] = text_input

    if not docs_dict:
        return [], np.array([]), [], [], "无文档内容", None

    _documents_cache = docs_dict

    preprocessor = DocumentPreprocessor(
        model_path=config.MODEL_PATH,
        encoder_name=config.ENCODER_NAME,
        device=config.DEVICE,
        compression_ratio=compression_ratio,
        batch_size=config.BATCH_SIZE,
        compressor=_compressor  # 使用当前切换的压缩器
    )

    full_text = "\n\n".join(docs_dict.values())
    total_tokens = count_tokens(_compressor.tokenizer, full_text)
    _strategy = CompressionStrategy(docs_dict)

    if preprocessing_mode == "auto":
        class Args:
            mode = "rag"
            global_compress = False
            use_token_merge = False
            use_learnable_merge = False

        args = Args()
        mode = _strategy.select_mode(args)
        if mode == "full_compress":
            preprocessing_mode_actual = "full_compress"
        elif mode == "global_compress_retrieve":
            preprocessing_mode_actual = "global_compress"
        else:
            preprocessing_mode_actual = "standard"
    else:
        preprocessing_mode_actual = preprocessing_mode

    print(f"\n🌐 预处理模式: {preprocessing_mode_actual}")

    progress(0.2, desc=f"执行 {preprocessing_mode_actual} 预处理...")

    if preprocessing_mode_actual == "full_compress":
        target = target_tokens or config.CONTEXT_WINDOW_SIZE
        original_tokens = count_tokens(_compressor.tokenizer, full_text)
        if original_tokens <= target:
            compressed_full = full_text
            compression_note = f"\n📝 注：原始文档 {original_tokens} tokens ≤ 目标 {target} tokens，无需压缩"
        else:
            compressed_full = _compressor.compress_to_target_tokens(full_text, target)
            compression_note = ""
        fragments = [compressed_full]
        vectors = np.array([])
        doc_ids = [0]
        doc_orders = [0]
        actual_tokens = count_tokens(_compressor.tokenizer, compressed_full)
        retrieval_hint = "\n⚠️ 注意：压缩比 ≤ 0.7，问答时将使用关键词锚定检索" if compression_ratio <= 0.7 else ""
        compression_stats = f"全文压缩直答模式 | 目标: {target} tokens | 实际: {actual_tokens} tokens{compression_note}{retrieval_hint}"

    elif preprocessing_mode_actual == "global_compress":
        target = target_tokens or config.CONTEXT_WINDOW_SIZE
        fragments, vectors, doc_ids, doc_orders = preprocessor.preprocess_global_compress(
            docs_dict,
            compression_ratio=compression_ratio,
            target_total_tokens=target,
            fragment_max_chars=getattr(config, 'FRAGMENT_SIZE', 384)
        )
        compression_stats = f"全局压缩+片段检索 | 目标: {target} tokens | 片段数: {len(fragments)}"

    elif preprocessing_mode_actual == "token_merge":
        fragments, vectors, doc_ids, doc_orders = preprocessor.preprocess_token_merge(
            docs_dict,
            compression_ratio=compression_ratio,
            sim_threshold=0.85,
            reduction_ratio=0.3
        )
        compression_stats = f"Token合并模式 | 压缩比: {compression_ratio:.0%} | 片段数: {len(fragments)}"

    elif preprocessing_mode_actual == "learnable_merge":
        fragments, vectors, doc_ids, doc_orders = preprocessor.preprocess_learnable_merge(
            docs_dict,
            compression_ratio=compression_ratio,
            use_cross_segment=False
        )
        compression_stats = f"可学习Token合并 | 压缩比: {compression_ratio:.0%} | 片段数: {len(fragments)}"

    elif preprocessing_mode_actual == "hierarchical":
        fragments, vectors, doc_ids, doc_orders = preprocessor.preprocess_hierarchical(
            docs_dict,
            compression_ratio=compression_ratio
        )
        compression_stats = f"层次化聚合 | 压缩比: {compression_ratio:.0%} | 片段数: {len(fragments)}"

    else:  # standard
        fragments, vectors, doc_ids, doc_orders = preprocessor.preprocess_independent(
            docs_dict,
            compression_ratio=compression_ratio
        )
        compression_stats = f"标准独立分段压缩 | 压缩比: {compression_ratio:.0%} | 片段数: {len(fragments)}"

    # 关键词检索器初始化
    if use_keyword_retrieval and fragments:
        _keyword_retriever = KeywordRetriever()
        _keyword_retriever.build_index(fragments)
        compression_stats += "\n🔑 关键词倒排索引已构建"
    else:
        _keyword_retriever = None

    # 保存压缩文件
    compressed_files_info = []
    compressed_file_path = None
    if fragments:
        try:
            import datetime, json
            batch_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_base_dir = os.path.join(config.PROJECT_ROOT, "data", "compressed_docs")
            batch_dir = os.path.join(output_base_dir, f"batch_{batch_timestamp}")
            os.makedirs(batch_dir, exist_ok=True)
            original_total_tokens = 0
            for doc_content in docs_dict.values():
                original_total_tokens += count_tokens(_compressor.tokenizer, doc_content)
            compressed_total_tokens = sum(count_tokens(_compressor.tokenizer, frag) for frag in fragments)

            if preprocessing_mode_actual == "full_compress":
                filename = f"full_compress_{batch_timestamp}.txt"
                file_path = os.path.join(batch_dir, filename)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(f"# 压缩文档信息\n{'='*80}\n")
                    f.write(f"批次号: {batch_timestamp}\n")
                    f.write(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"压缩比: {compression_ratio:.0%}\n")
                    f.write(f"预处理模式: {preprocessing_mode_actual}\n")
                    f.write(f"{'='*80}\n\n")
                    f.write(f"## Token统计\n")
                    f.write(f"原始文档数: {len(docs_dict)}\n")
                    f.write(f"原始总Token数: {original_total_tokens}\n")
                    f.write(f"压缩后Token数: {compressed_total_tokens}\n")
                    f.write(f"压缩率: {(1 - compressed_total_tokens/original_total_tokens)*100:.1f}%\n")
                    f.write(f"Token变化: {original_total_tokens} → {compressed_total_tokens} (减少 {original_total_tokens - compressed_total_tokens})\n\n")
                    f.write(f"{'='*80}\n\n## 压缩后内容\n\n")
                    f.write(fragments[0])
                compressed_files_info.append({
                    'path': file_path, 'name': filename,
                    'original_tokens': original_total_tokens,
                    'compressed_tokens': compressed_total_tokens
                })
            else:
                for doc_idx, (doc_name, doc_content) in enumerate(docs_dict.items()):
                    doc_fragments = [(i, f) for i, (f, did) in enumerate(zip(fragments, doc_ids)) if did == doc_idx]
                    if not doc_fragments:
                        continue
                    doc_original_tokens = count_tokens(_compressor.tokenizer, doc_content)
                    doc_compressed_tokens = sum(count_tokens(_compressor.tokenizer, frag) for _, frag in doc_fragments)
                    safe_name = "".join(c for c in doc_name if c.isalnum() or c in (' ', '-', '_')).strip()[:50]
                    filename = f"{safe_name}_{batch_timestamp}.txt"
                    file_path = os.path.join(batch_dir, filename)
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(f"# 压缩文档信息\n{'='*80}\n")
                        f.write(f"批次号: {batch_timestamp}\n")
                        f.write(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"原始文档: {doc_name}\n")
                        f.write(f"压缩比: {compression_ratio:.0%}\n")
                        f.write(f"预处理模式: {preprocessing_mode_actual}\n")
                        f.write(f"{'='*80}\n\n## Token统计\n")
                        f.write(f"原始Token数: {doc_original_tokens}\n")
                        f.write(f"压缩后Token数: {doc_compressed_tokens}\n")
                        f.write(f"压缩率: {(1 - doc_compressed_tokens/doc_original_tokens)*100:.1f}%\n")
                        f.write(f"Token变化: {doc_original_tokens} → {doc_compressed_tokens} (减少 {doc_original_tokens - doc_compressed_tokens})\n")
                        f.write(f"片段数量: {len(doc_fragments)}\n\n{'='*80}\n\n")
                        for frag_idx, (global_idx, frag) in enumerate(doc_fragments):
                            f.write(f"## 片段 {frag_idx + 1} (全局索引: {global_idx})\n")
                            f.write(frag)
                            f.write(f"\n\n{'-'*80}\n\n")
                    compressed_files_info.append({
                        'path': file_path, 'name': filename, 'doc_name': doc_name,
                        'original_tokens': doc_original_tokens, 'compressed_tokens': doc_compressed_tokens,
                        'fragments_count': len(doc_fragments)
                    })
            index_file = os.path.join(output_base_dir, "batch_index.json")
            if os.path.exists(index_file):
                with open(index_file, 'r', encoding='utf-8') as f:
                    index_data = json.load(f)
            else:
                index_data = {'batches': []}
            index_data['batches'].append({
                'batch_id': batch_timestamp,
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'compression_ratio': compression_ratio,
                'preprocessing_mode': preprocessing_mode_actual,
                'documents_count': len(docs_dict),
                'total_original_tokens': original_total_tokens,
                'total_compressed_tokens': compressed_total_tokens,
                'files': [{'filename': info['name'], 'original_tokens': info['original_tokens'], 'compressed_tokens': info['compressed_tokens']} for info in compressed_files_info]
            })
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, ensure_ascii=False, indent=2)
            print(f"✅ 压缩文档已保存至: {batch_dir}")
            if compressed_files_info:
                compressed_file_path = compressed_files_info[0]['path']
        except Exception as e:
            print(f"⚠️ 保存压缩文档失败: {e}")
            import traceback
            traceback.print_exc()
            compression_stats += "\n⚠️ 保存压缩文档失败"

    progress(1.0, desc="完成")
    return fragments, vectors, doc_ids, doc_orders, compression_stats, compressed_file_path


def answer_wrapper_advanced(question, fragments, vectors, doc_ids, doc_orders, config_state):
    global _qa_engine, _keyword_retriever

    compression_ratio = config_state.get("compression_ratio", 0.7)
    use_token_window = config_state.get("use_token_window", False)
    retrieval_mode = config_state.get("retrieval_mode", 0.0)
    context_window = config_state.get("context_window", config.CONTEXT_WINDOW_SIZE)

    if hasattr(_qa_engine, 'context_window'):
        _qa_engine.context_window = context_window

    total_tokens = sum(count_tokens(_compressor.tokenizer, frag) for frag in fragments) if fragments else 0

    if not fragments:
        try:
            hist = _qa_engine.history_manager.get_context_for_llm()
            prompt = f"基于以下对话历史回答问题。\n\n{hist}\n\n问题：{question}\n答案：" if hist else f"问题：{question}\n答案："
            answer, p_tok, a_tok = _qa_engine.llm_client.call(prompt)
            token_stats = f"输入 tokens: {p_tok}\n输出 tokens: {a_tok}\n总计: {p_tok + a_tok}\n模式：直接对话"
        except Exception as e:
            answer = f"生成答案时出错: {str(e)}"
            token_stats = f"错误：{str(e)}"
        _qa_engine.history_manager.add_turn(question, answer)
        return answer, token_stats, _qa_engine.history_manager.get_fallback_text()

    # 检索策略：根据压缩比选择不同的检索方式
    # 压缩比 >= 0.5：使用原有的RAG框架（混合检索）
    # 压缩比 < 0.5：根据压缩后的token数选择策略
    if compression_ratio >= 0.5:
        # 使用原有的RAG框架（混合检索）
        try:
            answer, info = _qa_engine.answer(
                question=question,
                documents=fragments,
                alpha=retrieval_mode,
                use_expansion=True,
                use_compression=True,
                use_smart_context=True
            )
            token_stats = f"输入 tokens: {info['token_stats']['prompt_tokens']}\n输出 tokens: {info['token_stats']['answer_tokens']}\n总计: {info['token_stats']['total']}\n模式：混合检索(RAG)"
        except Exception as e:
            answer = f"生成答案时出错: {str(e)}"
            token_stats = f"错误：{str(e)}"
            info = {"retrieved_chunks": []}
        _qa_engine.history_manager.add_turn(question, answer, retrieved_chunks=info.get('retrieved_chunks', []))
        return answer, token_stats, _qa_engine.history_manager.get_fallback_text()
    
    # 压缩比 < 0.5：根据压缩后的token数选择策略
    if total_tokens <= context_window:
        # 压缩后token数小于设定值，直接送入LLM
        context = fragments[0] if len(fragments) == 1 else "\n\n".join(fragments)
        try:
            answer, info = _qa_engine.answer_direct(question, context)
            token_stats = f"输入 tokens: {info['token_stats']['prompt_tokens']}\n输出 tokens: {info['token_stats']['answer_tokens']}\n总计: {info['token_stats']['total']}\n模式：直接压缩送入(压缩后{total_tokens}tokens ≤ {context_window})"
        except Exception as e:
            answer = f"生成答案时出错: {str(e)}"
            token_stats = f"错误：{str(e)}"
            info = {"retrieved_chunks": []}
        _qa_engine.history_manager.add_turn(question, answer, retrieved_chunks=info.get('retrieved_chunks', []))
        return answer, token_stats, _qa_engine.history_manager.get_fallback_text()
    else:
        # 压缩后token数大于设定值，使用关键词检索
        if _keyword_retriever is None:
            _keyword_retriever = KeywordRetriever()
            _keyword_retriever.build_index(fragments)
        retrieved = _keyword_retriever.retrieve(question, top_k=config.KEYWORD_RETRIEVAL_TOP_K)
        context = "\n\n".join(retrieved)
        try:
            answer, info = _qa_engine.answer_direct(question, context)
            token_stats = f"输入 tokens: {info['token_stats']['prompt_tokens']}\n输出 tokens: {info['token_stats']['answer_tokens']}\n总计: {info['token_stats']['total']}\n模式：关键词检索(压缩后{total_tokens}tokens > {context_window})"
        except Exception as e:
            answer = f"生成答案时出错: {str(e)}"
            token_stats = f"错误：{str(e)}"
            info = {"retrieved_chunks": retrieved}
        _qa_engine.history_manager.add_turn(question, answer, retrieved_chunks=info.get('retrieved_chunks', []))
        return answer, token_stats, _qa_engine.history_manager.get_fallback_text()


def update_model(backend_type, api_url, model_name, api_key):
    try:
        global _llm_client, _current_config, _qa_engine
        api_url = api_url.strip()
        model_name = model_name.strip()
        api_key = api_key.strip() if api_key else None
        if not api_url or not model_name:
            return "请填写完整的 API 地址和模型名称"
        if backend_type == "openai":
            if not api_url.endswith("/chat/completions"):
                api_url = api_url.rstrip("/") + "/chat/completions" if api_url.endswith("/") else api_url + "/chat/completions"
        new_llm = LLMClient(backend_type=backend_type, api_url=api_url, model_name=model_name, api_key=api_key, tokenizer=tokenizer)
        _llm_client = new_llm
        _qa_engine.llm_client = new_llm
        _current_config = {'backend_type': backend_type, 'api_url': api_url, 'model_name': model_name, 'api_key': api_key}
        return f"✅ 模型配置已更新！当前使用 {backend_type} - {model_name}"
    except Exception as e:
        return f"❌ 更新失败: {str(e)}"


def reset_history():
    """清空对话历史，返回空摘要和空Token统计"""
    if _qa_engine:
        _qa_engine.history_manager.clear()
    return "", ""


def on_compressor_model_change(model_type):
    try:
        switch_compressor_model(model_type, config.DEVICE)
        return f"✅ 压缩模型已切换为: {model_type}"
    except Exception as e:
        return f"❌ 切换失败: {str(e)}"


def apply_compression_config(compressor_model, compression_ratio, preprocessing_mode,
                             target_tokens, use_keyword, use_token_window,
                             retrieval_mode, context_window, relevance_weight):
    status = on_compressor_model_change(compressor_model)
    config_dict = {
        "compressor_model": compressor_model,
        "compression_ratio": compression_ratio,
        "preprocessing_mode": preprocessing_mode,
        "target_tokens": target_tokens,
        "use_keyword_retrieval": use_keyword,
        "use_token_window": use_token_window,
        "retrieval_mode": retrieval_mode,
        "context_window": context_window,
        "relevance_weight": relevance_weight,
    }
    if hasattr(_qa_engine, 'context_window'):
        _qa_engine.context_window = context_window
    return config_dict, status


def on_process_click(files, text_input, config_state, progress=gr.Progress()):
    compression_ratio = config_state.get("compression_ratio", 0.7)
    preprocessing_mode = config_state.get("preprocessing_mode", "auto")
    use_keyword = config_state.get("use_keyword_retrieval", False)
    target_tokens = config_state.get("target_tokens", config.CONTEXT_WINDOW_SIZE)
    use_token_window = config_state.get("use_token_window", False)

    fragments, vectors, doc_ids, doc_orders, stats, compressed_file = process_documents_advanced(
        files, text_input, compression_ratio, preprocessing_mode, use_keyword, target_tokens, progress
    )
    if _qa_engine:
        _qa_engine.history_manager.clear()

    total_tokens = sum(count_tokens(_compressor.tokenizer, frag) for frag in fragments) if fragments else 0

    if compression_ratio <= 0.7:
        retrieval_status = "🔍 低压缩比模式：关键词锚定检索已强制启用"
    else:
        window = config.CONTEXT_WINDOW_SIZE
        if use_token_window and total_tokens <= window:
            retrieval_status = f"⚡ 全文直答模式（窗口匹配已启用）：{total_tokens} tokens ≤ 窗口"
        elif use_token_window:
            retrieval_status = f"🔍 混合检索模式（窗口匹配已启用）：{total_tokens} tokens > 窗口"
        else:
            retrieval_status = f"🔍 混合检索模式（窗口匹配未启用）：{total_tokens} tokens"

    stats += f"\n{retrieval_status}"
    return fragments, vectors, doc_ids, doc_orders, stats, compressed_file, retrieval_status


def generate_visualization():
    try:
        tree = _qa_engine.history_manager.tree
        if not tree.info_nodes and not tree.qa_nodes:
            return "", "当前没有历史数据"
        output_path = os.path.join(config.PROJECT_ROOT, "tree_visualization.html")
        visualize_tree(tree, output_path, max_nodes=100)
        stats = _qa_engine.history_manager.get_vector_stats()
        stats_text = f"**统计信息**：信息节点 {len(tree.info_nodes)}，问答节点 {len(tree.qa_nodes)}"
        return f"file://{output_path}", stats_text
    except Exception as e:
        return "", f"❌ 生成失败: {str(e)}"


# ==================== Gradio 界面构建 ====================
with gr.Blocks(title="基于语义压缩的长文本问答系统") as demo:
    gr.Markdown("# 基于语义压缩的长文本问答系统")
    gr.Markdown("集成**语义优先级压缩**、**自适应检索**与**动态Token分配**，高效处理长文档问答。")

    # 全局状态
    compression_config_state = gr.State({
        "compressor_model": "mooscomp",
        "compression_ratio": 0.7,
        "preprocessing_mode": "auto",
        "target_tokens": config.CONTEXT_WINDOW_SIZE,
        "use_keyword_retrieval": False,
        "use_token_window": False,
        "retrieval_mode": 0.0,
        "context_window": config.CONTEXT_WINDOW_SIZE,
        "relevance_weight": 0.5
    })

    fragments_state = gr.State([])
    vectors_state = gr.State(np.array([]))
    doc_ids_state = gr.State([])
    doc_orders_state = gr.State([])
    compressed_file_state = gr.State(None)

    with gr.Tabs():
        # ================== 问答页面 ==================
        with gr.TabItem("💬 问答"):
            with gr.Row():
                # 左侧栏 (3)
                with gr.Column(scale=3, min_width=250):
                    gr.Markdown("### 📊 已处理文档信息")
                    processed_docs_output = gr.Textbox(label="", lines=4, interactive=False,
                                                       placeholder="处理文档后将显示压缩统计信息...")
                    token_stats_output = gr.Textbox(label="📈 Token 统计", lines=5, interactive=False)

                    with gr.Accordion("⚙️ 模型配置", open=False):
                        backend_radio = gr.Radio(choices=["ollama", "openai"], value="ollama", label="后端类型")
                        api_url_input = gr.Textbox(label="API 地址", value=config.OLLAMA_URL)
                        model_name_input = gr.Textbox(label="模型名称", value=config.MODEL_NAME)
                        api_key_input = gr.Textbox(label="API Key（可选）", type="password")
                        update_btn = gr.Button("更新模型配置", variant="secondary")
                        config_status = gr.Textbox(label="配置状态", interactive=False)

                    retrieval_status = gr.Textbox(label="🔍 当前检索策略", interactive=False,
                                                  value="等待处理文档...")
                    reset_history_btn = gr.Button("🗑️ 清空对话历史", variant="secondary")

                # 右侧栏 (7)
                with gr.Column(scale=7):
                    with gr.Row():
                        input_mode_radio = gr.Radio(choices=["📁 上传文件", "✏️ 输入文本"],
                                                    value="📁 上传文件", label="输入方式", scale=1)
                    file_input = gr.File(label="上传文档（支持多文件）", file_count="multiple", visible=True)
                    text_input = gr.Textbox(label="或直接输入文本内容", lines=6, visible=False,
                                            placeholder="在此输入文本内容...")

                    with gr.Row():
                        process_btn = gr.Button("📄 处理文档", variant="primary", size="lg")
                        compressed_file_output = gr.File(label="📥 下载压缩文档", visible=False)

                    question_input = gr.Textbox(label="问题", lines=4, placeholder="输入你的问题...")
                    ask_btn = gr.Button("💡 提问", variant="primary", size="lg")
                    answer_output = gr.Textbox(label="答案", lines=10, interactive=False)

            # 仅在本标签页内可用的组件事件（无需跨标签页）
            process_btn.click(
                fn=on_process_click,
                inputs=[file_input, text_input, compression_config_state],
                outputs=[fragments_state, vectors_state, doc_ids_state, doc_orders_state,
                         processed_docs_output, compressed_file_state, retrieval_status]
            ).then(
                fn=lambda f: gr.update(visible=True, value=f) if f else gr.update(visible=False),
                inputs=[compressed_file_state],
                outputs=[compressed_file_output]
            )

            update_btn.click(fn=update_model, inputs=[backend_radio, api_url_input, model_name_input, api_key_input],
                             outputs=[config_status])

            def toggle_input_mode(mode):
                if mode == "📁 上传文件":
                    return gr.update(visible=True), gr.update(visible=False)
                else:
                    return gr.update(visible=False), gr.update(visible=True)

            input_mode_radio.change(fn=toggle_input_mode, inputs=[input_mode_radio], outputs=[file_input, text_input])

            gr.Markdown("**说明**：请在「压缩配置」页面调整参数，然后点击「处理文档」。")

        # ================== 压缩配置页面 ==================
        with gr.TabItem("🗜️ 压缩配置"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## 语义压缩参数")
                    compressor_model_dropdown = gr.Dropdown(
                        choices=["mooscomp", "llmlingua2", "word_priority", "hybrid"],
                        value="mooscomp",
                        label="压缩模型",
                        info="mooscomp: Token级二分类 | llmlingua2: LLMLingua2 | word_priority: 词级优先级 | hybrid: 混合压缩"
                    )
                    compression_ratio_slider = gr.Slider(0.1, 1.0, 0.7, step=0.05, label="语义保留比率")
                    preprocessing_mode_dropdown = gr.Dropdown(
                        choices=["auto", "standard", "global_compress", "token_merge", "learnable_merge", "hierarchical"],
                        value="auto", label="语义压缩策略",
                        info="auto: 自适应选择；standard: 独立分段；global_compress: 全局压缩后切分"
                    )
                    target_tokens_number = gr.Number(label="目标Token数（仅global_compress/full_compress有效）",
                                                     value=config.CONTEXT_WINDOW_SIZE, precision=0)
                    use_keyword_checkbox = gr.Checkbox(label="启用关键词锚定检索", value=False)
                    use_token_window_checkbox = gr.Checkbox(label="启用Token窗口匹配（压缩比>0.7时生效）",
                                                            value=False,
                                                            info="启用后：压缩后Token≤窗口→直答；否则→检索")
                with gr.Column():
                    gr.Markdown("## 检索配置")
                    retriever_slider = gr.Slider(-1.0, 1.0, 0.0, step=0.1, label="检索模式 (-1:关键词 0:混合 1:语义)")
                    context_window_slider = gr.Slider(1024, 8192, config.CONTEXT_WINDOW_SIZE, step=128, label="上下文窗口")
                    relevance_weight_slider = gr.Slider(0.0, 1.0, 0.5, step=0.05, label="历史相关性权重")

            with gr.Row():
                apply_btn = gr.Button("📌 应用全部配置", variant="primary")
                config_apply_status = gr.Textbox(label="配置状态", interactive=False, lines=1)

            apply_btn.click(
                fn=apply_compression_config,
                inputs=[compressor_model_dropdown, compression_ratio_slider, preprocessing_mode_dropdown,
                        target_tokens_number, use_keyword_checkbox, use_token_window_checkbox,
                        retriever_slider, context_window_slider, relevance_weight_slider],
                outputs=[compression_config_state, config_apply_status]
            )

            gr.Markdown("""
            **💡 说明**：
            - 调整参数后请点击“应用全部配置”使其生效。
            - 检索策略会按压缩比和Token窗口自动选择，此处仅设定基础检索偏好。
            """)

        # ================== 对话历史页面 ==================
        with gr.TabItem("📜 对话历史"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📜 对话历史摘要")
                    history_summary_output = gr.Textbox(label="", lines=10, interactive=False,
                                                        placeholder="暂无对话历史...", autoscroll=False)
                with gr.Column(scale=1):
                    gr.Markdown("### 🌲 多源信息树可视化")
                    visualize_btn = gr.Button("生成树可视化", variant="primary")
                    tree_stats_output = gr.Markdown()
                    visualization_output = gr.HTML()

            visualize_btn.click(fn=generate_visualization, inputs=[],
                                outputs=[visualization_output, tree_stats_output])

    # ==================== 全局事件绑定（跨标签页） ====================
    # 因为 reset_history_btn 在问答页面，而 history_summary_output 在对话历史页面，
    # 必须在所有组件定义完毕后进行绑定。
    reset_history_btn.click(fn=reset_history, inputs=[],
                            outputs=[history_summary_output, token_stats_output])

    ask_btn.click(
        fn=answer_wrapper_advanced,
        inputs=[question_input, fragments_state, vectors_state, doc_ids_state, doc_orders_state,
                compression_config_state],
        outputs=[answer_output, token_stats_output, history_summary_output]
    )


if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)