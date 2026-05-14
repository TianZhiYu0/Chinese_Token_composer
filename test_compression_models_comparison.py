#!/usr/bin/env python3
"""
四种压缩模型对比测试脚本
==========================
测试压缩模型：
- LLMLingua-2 (token级压缩模型)
- MOOSComp (token级二分类模型)
- WordPriority (词级优先级模型)
- Hybrid (MOOSComp + WordPriority 混合模型)

测试流程：
1. 加载测试集 test_10k.json
2. 将所有文档合并成一个整体
3. 使用四种压缩模型分别在不同压缩比下进行压缩
4. 将压缩后的文档 + 每个问题输入到 LLM 生成答案
5. 使用 DeepSeek API 对答案进行评分
6. 对比不同模型的问答效果并生成测试报告

使用方法：
    python test_compression_models_comparison.py \
      --test_file Model_traing/test_data/test_data/test_10k.json \
      --compression_ratios 0.7 0.5 0.3 \
      --output eval_results/compression_models_comparison.json
"""
import os
import sys
import json
import argparse
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

import config
from core.compression.compressor import HardCompressor
from core.compression.word_priority_compressor import WordPriorityCompressor
from core.compression.hybrid_compressor import HybridCompressor
from core.model.llm_client import LLMClient
from core.utils.utils import split_sentences

DEEPSEEK_API_KEY = "sk-55a17ff86963473499e86644dc89152d"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

def call_with_retry(func, max_retries=3, *args, **kwargs):
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            print(f"重试 {attempt+1}/{max_retries}: {e}")
            continue

def parse_scores(response: str) -> Dict[str, float]:
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
        scores = {}
        for key in ["准确性", "完整性", "相关性"]:
            pattern = rf'["\']?{key}["\']?\s*:\s*(\d+(?:\.\d+)?)'
            m = re.search(pattern, response)
            if m:
                scores[key] = float(m.group(1))
        if len(scores) == 3:
            return scores
        print(f"警告：无法解析评分响应: {response[:200]}")
        return {"准确性": 0, "完整性": 0, "相关性": 0}

def build_scoring_prompt(question: str, reference: str, answer: str) -> str:
    return f"""你是一个专业的答案质量评估专家。请根据以下标准对系统给出的答案进行评分（0-100分）：
1. 准确性：答案是否与标准答案一致，无事实错误。
2. 完整性：是否覆盖了标准答案中的关键信息。
3. 相关性：答案是否直接回应问题，无偏离。

问题：{question}
标准答案：{reference}
系统答案：{answer}

请以 JSON 格式输出评分，例如：
{{"准确性": 85, "完整性": 90, "相关性": 95}}
只输出 JSON，不要有其他内容。"""

def build_qa_prompt(question: str, compressed_context: str) -> str:
    return f"""请根据以下参考资料回答问题：

参考资料：
{compressed_context}

问题：{question}

请仅基于参考资料中的信息回答问题，不要编造信息。"""

def compress_text(
    text: str,
    compression_ratio: float,
    compressor,
    compressor_type: str,
    chunk_size: int = 100
) -> Tuple[str, int, int]:
    """通用压缩函数，支持不同类型压缩器"""
    sentences = split_sentences(text)
    if not sentences:
        return "", 0, 0
    
    compressed_sentences = []
    total_batches = (len(sentences) + chunk_size - 1) // chunk_size
    
    for i in range(0, len(sentences), chunk_size):
        chunk = sentences[i:i+chunk_size]
        try:
            if compressor_type in ["llmlingua2", "mooscomp"]:
                compressed_chunk = compressor.compress_batch(chunk, compression_ratio=compression_ratio)
            elif compressor_type == "word_priority":
                compressed_chunk = compressor.compress_batch(chunk, compression_ratio=compression_ratio)
            elif compressor_type == "hybrid":
                compressed_chunk = compressor.compress_batch(chunk, compression_ratio=compression_ratio)
            else:
                compressed_chunk = chunk
            compressed_sentences.extend(compressed_chunk)
        except Exception as e:
            print(f"⚠️  {compressor_type} 分段压缩失败 ({i//chunk_size + 1}/{total_batches}): {e}")
            compressed_sentences.extend(chunk)  # 保留原始内容
    
    compressed_text = "".join(compressed_sentences)
    original_length = len(text)
    compressed_length = len(compressed_text)
    return compressed_text, original_length, compressed_length

def main():
    parser = argparse.ArgumentParser(description="四种压缩模型对比测试")
    parser.add_argument("--test_file", required=True, help="测试集 JSON 文件")
    parser.add_argument("--output", default="eval_results/compression_models_comparison.json",
                       help="输出结果文件（会自动根据压缩比生成文件名）")
    parser.add_argument("--compression_ratios", nargs="+", type=float, default=[0.7, 0.5, 0.3],
                       help="测试的压缩比列表")
    parser.add_argument("--gen_backend", default="ollama", choices=["ollama", "openai"])
    parser.add_argument("--gen_api_url", default=None, help="生成答案的 API 地址")
    parser.add_argument("--gen_model", default=None, help="生成答案的模型名称")
    parser.add_argument("--gen_api_key", default=None, help="生成答案的 API Key")
    args = parser.parse_args()

    # 根据压缩比参数生成动态输出文件名
    if args.compression_ratios:
        ratios_str = "_".join([str(int(r * 100)) for r in args.compression_ratios])
        output_dir = os.path.dirname(args.output)
        output_base = os.path.basename(args.output)
        # 移除扩展名
        if output_base.endswith('.json'):
            output_name = output_base[:-5]
        else:
            output_name = output_base
        # 添加压缩比后缀
        args.output = os.path.join(output_dir, f"{output_name}_{ratios_str}.json")

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    print("="*80)
    print(" 四种压缩模型对比测试")
    print(" (LLMLingua-2 | MOOSComp | WordPriority | Hybrid)")
    print("="*80)
    print(f"测试文件: {args.test_file}")
    print(f"测试压缩比: {args.compression_ratios}")
    print()

    print("📋 加载测试集...")
    with open(args.test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    all_documents = test_data["documents"]
    all_qa_pairs = test_data["qa_pairs"]
    print(f"   总文档数: {len(all_documents)}")
    print(f"   总 QA 对数: {len(all_qa_pairs)}")
    print(f"   总长度: {test_data.get('metadata', {}).get('actual_total_length', 'N/A')}")
    print()

    print("📄 合并所有文档...")
    combined_text = ""
    for doc_name, doc_text in all_documents.items():
        combined_text += doc_text + "\n"
    original_combined_length = len(combined_text)
    print(f"   合并后总长度: {original_combined_length:,} 字符")
    print()

    print("🔧 初始化四个压缩模型...")
    MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
    MOOSCOMP_MODEL_PATH = os.path.join(MODEL_DIR, "compression_bert_mooscomp_news")
    LLMLINGUA_MODEL_PATH = os.path.join(MODEL_DIR, "llmlingua-2-bert-base-multilingual-cased-meetingbank")
    WORDS_PRIORITY_MODEL_PATH = os.path.join(MODEL_DIR, "word_priority_model_mixed", "best_model.pt")

    # 1. LLMLingua-2 压缩器
    llmlingua_compressor = HardCompressor(
        model_path=LLMLINGUA_MODEL_PATH,
        model_type="llmlingua2",
        device=config.DEVICE
    )
    print("   ✅ LLMLingua-2 压缩器初始化完成")

    # 2. MOOSComp 压缩器
    mooscomp_compressor = HardCompressor(
        model_path=MOOSCOMP_MODEL_PATH,
        model_type="mooscomp",
        device=config.DEVICE
    )
    print("   ✅ MOOSComp 压缩器初始化完成")

    # 3. WordPriority 压缩器
    word_priority_compressor = WordPriorityCompressor(
        bert_model_path=MOOSCOMP_MODEL_PATH,
        priority_model_path=WORDS_PRIORITY_MODEL_PATH,
        device=config.DEVICE
    )
    print("   ✅ WordPriority 压缩器初始化完成")

    # 4. Hybrid 混合压缩器
    hybrid_compressor = HybridCompressor(
        bert_model_path=MOOSCOMP_MODEL_PATH,
        word_priority_model_path=WORDS_PRIORITY_MODEL_PATH,
        mooscomp_model_path=MOOSCOMP_MODEL_PATH,
        device=config.DEVICE,
        dynamic_alpha=True,
        fixed_alpha=0.85
    )
    print("   ✅ Hybrid 混合压缩器初始化完成")
    print()

    # 压缩器配置
    compressors = {
        "llmlingua2": llmlingua_compressor,
        "mooscomp": mooscomp_compressor,
        "word_priority": word_priority_compressor,
        "hybrid": hybrid_compressor
    }

    print("🤖 初始化 LLM 客户端...")
    gen_api_url = args.gen_api_url or (config.OLLAMA_URL if args.gen_backend == "ollama" else None)
    gen_model = args.gen_model or (config.MODEL_NAME if args.gen_backend == "ollama" else None)
    print(f"   生成模型: {gen_model} ({args.gen_backend})")

    gen_llm = LLMClient(
        backend_type=args.gen_backend,
        api_url=gen_api_url,
        model_name=gen_model,
        api_key=args.gen_api_key if args.gen_backend == "openai" else None,
        tokenizer=mooscomp_compressor.tokenizer
    )

    critic_llm = LLMClient(
        backend_type="openai",
        api_url=DEEPSEEK_API_URL,
        model_name=DEEPSEEK_MODEL,
        api_key=DEEPSEEK_API_KEY,
        tokenizer=mooscomp_compressor.tokenizer
    )
    print("   ✅ LLM 客户端初始化完成")
    print()

    print("📦 预压缩合并后的文档...")
    compressed_results = {}
    
    # 根据显存大小调整分段大小
    chunk_size = 100  # 每批处理100个句子，减少内存占用

    for ratio in args.compression_ratios:
        ratio_key = f"ratio_{int(ratio*100)}"
        compressed_results[ratio_key] = {}

        for strategy, compressor in compressors.items():
            print(f"   处理 {strategy} {ratio:.0%}...")
            comp_text, _, comp_len = compress_text(
                combined_text, ratio, compressor, strategy, chunk_size=chunk_size
            )
            compressed_results[ratio_key][strategy] = {
                "text": comp_text,
                "compressed_length": comp_len,
                "actual_ratio": comp_len / original_combined_length if original_combined_length > 0 else 0
            }
            print(f"   {strategy} {ratio:.0%}: {original_combined_length:,} -> {comp_len:,} "
                  f"({comp_len/original_combined_length*100:.1f}%)")
    print()

    results = []
    total_tests = len(all_qa_pairs) * len(args.compression_ratios) * len(compressors)
    current_test = 0

    print("="*80)
    print(f"🎯 开始测试（共 {total_tests} 个测试用例）")
    print("="*80)
    print()

    for qa_idx, qa_item in enumerate(all_qa_pairs):
        question = qa_item["question"]
        ref_answer = qa_item["answer"]
        doc_name = qa_item["document"]
        difficulty = qa_item.get("difficulty", "unknown")
        qa_type = qa_item.get("type", "unknown")

        print(f"[{qa_idx+1}/{len(all_qa_pairs)}] 问题: {question[:60]}...")
        print(f"   文档: {doc_name[:50]}...")
        print(f"   难度: {difficulty}, 类型: {qa_type}")

        for ratio in args.compression_ratios:
            ratio_key = f"ratio_{int(ratio*100)}"

            for strategy in compressors.keys():
                current_test += 1
                comp_data = compressed_results[ratio_key][strategy]
                comp_text = comp_data["text"]
                actual_ratio = comp_data["actual_ratio"]
                comp_length = comp_data["compressed_length"]

                print(f"  [{current_test}/{total_tests}] {strategy} {ratio:.0%} "
                      f"(实际={actual_ratio:.0%}, {comp_length:,} 字)...", end="", flush=True)

                qa_prompt = build_qa_prompt(question, comp_text)

                def generate_answer():
                    return gen_llm.call(qa_prompt)

                try:
                    answer, prompt_tokens, answer_tokens = call_with_retry(
                        generate_answer, max_retries=2
                    )
                except Exception as e:
                    answer = f"生成失败: {str(e)}"
                    prompt_tokens = 0
                    answer_tokens = 0
                    print(f" ❌ 生成失败: {e}")
                else:
                    def score_answer():
                        scoring_prompt = build_scoring_prompt(question, ref_answer, answer)
                        response, _, _ = critic_llm.call(scoring_prompt)
                        return parse_scores(response)

                    try:
                        scores = call_with_retry(score_answer, max_retries=2)
                    except Exception as e:
                        print(f" ⚠️ 评分失败: {e}")
                        scores = {"准确性": 0, "完整性": 0, "相关性": 0}

                    overall = sum(scores.values()) / len(scores)
                    print(f" ✅ 得分: {overall:.1f}")

                    result = {
                        "question_index": qa_idx,
                        "question": question,
                        "reference_answer": ref_answer,
                        "document": doc_name,
                        "difficulty": difficulty,
                        "type": qa_type,
                        "compression_strategy": strategy,
                        "target_ratio": ratio,
                        "actual_ratio": actual_ratio,
                        "compressed_length": comp_length,
                        "original_combined_length": original_combined_length,
                        "system_answer": answer,
                        "scores": scores,
                        "overall_score": overall,
                        "token_stats": {
                            "prompt_tokens": prompt_tokens,
                            "answer_tokens": answer_tokens,
                            "total": prompt_tokens + answer_tokens
                        }
                    }
                    results.append(result)
        print()

    print("="*80)
    print("📊 统计分析")
    print("="*80)

    group_stats = defaultdict(list)
    for r in results:
        key = (r["compression_strategy"], r["target_ratio"])
        group_stats[key].append(r)

    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "test_file": args.test_file,
        "selected_documents": list(all_documents.keys()),
        "selected_qa_count": len(all_qa_pairs),
        "original_combined_length": original_combined_length,
        "compression_ratios": args.compression_ratios,
        "group_summary": {},
        "total_results": results
    }

    # 按压缩比组织结果，便于对比
    for ratio in args.compression_ratios:
        print(f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"📈 压缩比 {ratio:.0%} 下各模型对比")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        ratio_results = []
        for strategy in compressors.keys():
            key = (strategy, ratio)
            items = group_stats.get(key, [])
            if not items:
                continue
            
            avg_scores = {}
            for score_key in ["准确性", "完整性", "相关性"]:
                values = [r["scores"].get(score_key, 0) for r in items]
                avg_scores[score_key] = sum(values) / len(values) if values else 0
            avg_overall = sum(r["overall_score"] for r in items) / len(items) if items else 0
            avg_tokens = sum(r["token_stats"]["total"] for r in items) / len(items) if items else 0
            
            # 获取实际压缩比
            actual_ratio = items[0]["actual_ratio"] if items else 0
            
            ratio_results.append({
                "strategy": strategy,
                "target_ratio": ratio,
                "actual_ratio": actual_ratio,
                "sample_count": len(items),
                "avg_scores": avg_scores,
                "avg_overall": avg_overall,
                "avg_tokens": avg_tokens
            })
            
            summary["group_summary"][f"{strategy}_{int(ratio*100)}"] = {
                "strategy": strategy,
                "compression_ratio": ratio,
                "actual_ratio": actual_ratio,
                "sample_count": len(items),
                "average_scores": avg_scores,
                "average_overall_score": avg_overall,
                "average_total_tokens": avg_tokens
            }
        
        # 按综合得分排序并打印
        ratio_results.sort(key=lambda x: x["avg_overall"], reverse=True)
        
        for i, res in enumerate(ratio_results):
            print(f"\n{i+1}. {res['strategy']}:")
            print(f"   ├─ 目标压缩比: {res['target_ratio']:.0%}")
            print(f"   ├─ 实际压缩比: {res['actual_ratio']:.1%}")
            print(f"   ├─ 样本数: {res['sample_count']}")
            print(f"   ├─ 平均得分:")
            print(f"   │   ├─ 准确性: {res['avg_scores']['准确性']:.1f}")
            print(f"   │   ├─ 完整性: {res['avg_scores']['完整性']:.1f}")
            print(f"   │   └─ 相关性: {res['avg_scores']['相关性']:.1f}")
            print(f"   ├─ 综合得分: {res['avg_overall']:.1f}")
            print(f"   └─ 平均 Token 数: {res['avg_tokens']:.0f}")

    # 生成总体对比报告
    print("\n" + "="*80)
    print("📊 总体对比报告")
    print("="*80)
    
    # 按模型汇总
    model_summary = defaultdict(list)
    for (strategy, ratio), items in group_stats.items():
        if items:
            avg_overall = sum(r["overall_score"] for r in items) / len(items)
            model_summary[strategy].append((ratio, avg_overall))
    
    print("\n各模型在不同压缩比下的综合得分：")
    print(f"{'模型':<15} {'70%压缩':<10} {'50%压缩':<10} {'30%压缩':<10}")
    print("-" * 50)
    
    for strategy in compressors.keys():
        scores = model_summary.get(strategy, [])
        score_dict = {r[0]: r[1] for r in scores}
        print(f"{strategy:<15} "
              f"{score_dict.get(0.7, 0):<10.1f} "
              f"{score_dict.get(0.5, 0):<10.1f} "
              f"{score_dict.get(0.3, 0):<10.1f}")

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 测试完成！")
    print(f"💾 结果已保存: {args.output}")
    print("="*80)

if __name__ == "__main__":
    main()
