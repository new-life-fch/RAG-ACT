import os
import argparse
import json
import csv
import pickle
from typing import List, Tuple, Dict, Optional

import torch
import numpy as np
from tqdm import tqdm
from einops import rearrange

import llama
from baukit import TraceDict
from llama_utils import (
    _load_nq_jsonl,
    _build_messages_input,
    get_interventions_dict,
    evaluate_nq_em_f1,
)


# 模型路径映射（与现有脚本保持一致）
HF_NAMES = {
    'llama2_chat_7B': '/root/shared-nvme/RAG-llm/models/Llama-2-7b-chat-hf',
    'llama3_8B_instruct': '/root/shared-nvme/RAG-llm/models/Llama-3.1-8B-Instruct',
}


def build_nq_generation_inputs(
    jsonl_path: str,
    tokenizer,
    max_samples: int = None,
    max_docs: int = None,
    use_chat_template: bool = True,
    sample_size: Optional[int] = None,
    sample_seed: int = 2025,
) -> Tuple[List[torch.Tensor], List[List[str]]]:
    """
    为 NQ 数据生成阶段构造输入：系统 + 用户消息（不包含助手答案）。
    返回：
    - inputs: 列表，元素为 `torch.Tensor` 的 `input_ids`
    - gold_answers_list: 列表，元素为正确答案字符串列表（用于 EM/F1）
    """
    entries = _load_nq_jsonl(jsonl_path, max_samples=max_samples)

    if sample_size is not None and sample_size > 0:
        rng = np.random.RandomState(sample_seed)
        idxs = rng.choice(len(entries), size=min(sample_size, len(entries)), replace=False)
        entries = [entries[i] for i in idxs]

    inputs: List[torch.Tensor] = []
    gold_answers_list: List[List[str]] = []

    for i, ex in enumerate(entries):
        if not all(k in ex for k in ["query", "answers", "retrieve_snippets"]):
            raise ValueError(f"JSONL 第 {i} 条记录缺少 query/answers/retrieve_snippets 字段")

        question = ex["query"]
        answers = ex["answers"]
        snippets = ex["retrieve_snippets"]

        docs_texts = []
        for j, snip in enumerate(snippets):
            if max_docs is not None and j >= max_docs:
                break
            text = snip.get("text", "")
            if isinstance(text, str) and len(text.strip()):
                docs_texts.append(text.strip())

        docs_block = "\n\n".join([f"Document {k+1}: {d}" for k, d in enumerate(docs_texts)])
        system_prompt = (
            "Answer the question based on the given document. "
            "Provide only the most direct and concise answer. Do not include explanations, full sentences, or additional context. "
            "Just give the key information that directly answers the question.\n\n"
            "Example:\n"
            "Question: when does nathan make it to the nba\n"
            "Answer: season 6 finale\n\n"
            f"The following are given documents.\n\n{docs_block}"
        )
        user_prompt = f"Question: {question}\nAnswer:"

        input_ids = _build_messages_input(tokenizer, system_prompt, user_prompt, assistant_content=None, use_chat_template=use_chat_template)
        inputs.append(input_ids)
        gold_answers_list.append(list(answers))

    return inputs, gold_answers_list


def load_scores_csv(path: str) -> List[Tuple[int, int, float]]:
    """读取 CSV，返回 (layer, head, score) 列表。"""
    items: List[Tuple[int, int, float]] = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            try:
                layer = int(row[0])
                head = int(row[1])
                score = float(row[2])
                items.append((layer, head, score))
            except Exception:
                # 跳过标题或异常行
                continue
    return items


def infer_LH_from_scores(scores: List[Tuple[int, int, float]]) -> Tuple[int, int]:
    L = max([l for l, _, _ in scores]) + 1 if scores else 0
    H = max([h for _, h, _ in scores]) + 1 if scores else 0
    return L, H


def make_selection_strategies(
    scores: List[Tuple[int, int, float]],
    L: int,
    H: int,
    val_accs: Optional[np.ndarray] = None,
) -> Dict[str, List[Tuple[int, int]]]:
    """
    构造多种头选择方案（按指定集合）：
    - 分数阈值：score>= {0.5,0.6,0.7,0.8,0.9}
    - 层区间：layers_{0_10, 10_20, 20_31, 0_20, 10_31, 0_31}
      区间采用零基索引，右端为 32 以包含第 31 层：例如 20_31 -> [20, min(32, L))。
    - 全部头：all_heads
    - 全局 Top-k：topk_{64,128,256,512}_by_score（按 CSV 分数降序）
    - 分层 Top-m：per_layer_top_{1,2,4,8,16}
    返回字典：策略名 -> (layer, head) 列表
    """
    by_score = sorted(scores, key=lambda x: x[2], reverse=True)
    strategies: Dict[str, List[Tuple[int, int]]] = {}

    # 分数阈值
    for thr in [0.5, 0.6, 0.7, 0.8, 0.9]:
        strategies[f'score_ge_{thr}'] = [(l, h) for (l, h, s) in scores if s >= thr]

    # 层区间（右端做 32 层兼容处理）
    def clamp_end(e: int) -> int:
        return min(e, L)

    strategies['layers_0_10'] = [(l, h) for (l, h, s) in scores if 0 <= l < clamp_end(10)]
    strategies['layers_10_20'] = [(l, h) for (l, h, s) in scores if 10 <= l < clamp_end(20)]
    strategies['layers_20_31'] = [(l, h) for (l, h, s) in scores if 20 <= l < clamp_end(32)]
    strategies['layers_0_20'] = [(l, h) for (l, h, s) in scores if 0 <= l < clamp_end(20)]
    strategies['layers_10_31'] = [(l, h) for (l, h, s) in scores if 10 <= l < clamp_end(32)]
    strategies['layers_0_31'] = [(l, h) for (l, h, s) in scores if 0 <= l < clamp_end(32)]

    # 全部头
    strategies['all_heads'] = [(l, h) for (l, h, s) in scores]

    # 全局 top-k（按分数降序）
    K_set = [64, 128, 256, 512]
    total = len(by_score)
    for k in K_set:
        kk = min(k, total)
        strategies[f'topk_{kk}_by_score'] = [(l, h) for (l, h, s) in by_score[:kk]]

    # 分层 top-m（每层按分数取前 m 个）
    grouped: Dict[int, List[Tuple[int, int, float]]] = {l: [] for l in range(L)}
    for (l, h, s) in scores:
        grouped.setdefault(l, []).append((l, h, s))
    for m in [1, 2, 4, 8, 16]:
        sel: List[Tuple[int, int]] = []
        for l in range(L):
            heads_sorted = sorted(grouped.get(l, []), key=lambda x: x[2], reverse=True)
            sel.extend([(l, h) for (_, h, _) in heads_sorted[:m]])
        strategies[f'per_layer_top_{m}'] = sel

    return strategies


def clean_answer_text(gen_str: str) -> str:
    s = gen_str.strip()
    if 'Q:' in s:
        s = s.split('Q:')[0].strip()
    if 'Answer:' in s:
        try:
            s = s.split('Answer:')[1].strip()
        except Exception:
            pass
    s = s.split('\n')[0].strip()
    return s


def run_baseline(model, tokenizer, device, inputs: List[torch.Tensor], gold_answers_list: List[List[str]], max_new_tokens: int) -> Tuple[List[str], Dict[str, float]]:
    preds_baseline: List[str] = []
    with torch.no_grad():
        for input_ids in tqdm(inputs, desc='baseline_generate'):
            input_ids = input_ids.to(device)
            gen_tokens = model.generate(
                input_ids,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
            )[:, input_ids.shape[-1]:]
            gen_str = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
            preds_baseline.append(clean_answer_text(gen_str))
    em_b, f1_b = evaluate_nq_em_f1(preds_baseline, gold_answers_list)
    return preds_baseline, {"EM": em_b, "F1": f1_b}


def run_intervention(
    model,
    tokenizer,
    device,
    inputs: List[torch.Tensor],
    gold_answers_list: List[List[str]],
    alpha: float,
    strategy_name: str,
    top_heads: List[Tuple[int, int]],
    probes: List,
    tuning_headwise: np.ndarray,
    head_dim: int,
    num_heads: Optional[int],
    use_probe_factor: bool,
    val_accs: Optional[np.ndarray],
    max_new_tokens: int,
) -> Tuple[List[str], Dict[str, float]]:
    B, L, HD = tuning_headwise.shape
    if num_heads is None:
        if HD % head_dim != 0:
            raise ValueError('无法从特征维推断 num_heads，请显式传入 --num_heads')
        num_heads_calc = HD // head_dim
    else:
        num_heads_calc = num_heads

    tuning_sep = rearrange(tuning_headwise, 'b l (h d) -> b l h d', h=num_heads_calc, d=head_dim)

    probe_score_map = val_accs if use_probe_factor and (val_accs is not None) else None
    interventions = get_interventions_dict(
        top_heads,
        probes,
        tuning_sep,
        num_heads_calc,
        use_center_of_mass=False,
        use_random_dir=False,
        com_directions=None,
        probe_score_map=probe_score_map,
    )

    def lt_modulated_vector_add(head_output, layer_name, start_edit_location='lt'):
        h_out = rearrange(head_output, 'b s (h d) -> b s h d', h=num_heads_calc)
        start_idx = -1 if start_edit_location == 'lt' else -1
        for head, direction, proj_val_std, probe_factor in interventions[layer_name]:
            direction_to_add = torch.tensor(direction).to(h_out.device)
            strength = alpha * proj_val_std * probe_factor
            if start_idx == -1:
                h_out[:, -1, head, :] += strength * direction_to_add
            else:
                h_out[:, start_idx:, head, :] += strength * direction_to_add
        return rearrange(h_out, 'b s h d -> b s (h d)')

    preds: List[str] = []
    layers_to_intervene = list(interventions.keys())
    with torch.no_grad():
        for input_ids in tqdm(
            inputs,
            desc=f'intervene[{strategy_name}] alpha={alpha} pf={int(use_probe_factor)}'
        ):
            input_ids = input_ids.to(device)
            with TraceDict(model, layers_to_intervene, edit_output=lt_modulated_vector_add):
                gen_tokens = model.generate(
                    input_ids,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id,
                )[:, input_ids.shape[-1]:]
            gen_str = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
            preds.append(clean_answer_text(gen_str))

    em_i, f1_i = evaluate_nq_em_f1(preds, gold_answers_list)
    return preds, {"EM": em_i, "F1": f1_i}


def main():
    parser = argparse.ArgumentParser(description='NQ 干预超参数搜索')
    parser.add_argument('--model_name', type=str, default='llama2_chat_7B')
    parser.add_argument('--dataset_path', type=str, default='/root/shared-nvme/RAG-llm/RAG/data/test.jsonl')
    parser.add_argument('--use_chat_template', action='store_true')
    parser.add_argument('--max_docs', type=int, default=None)
    parser.add_argument('--sample_size', type=int, default=100)
    parser.add_argument('--sample_seed', type=int, default=2025)
    parser.add_argument('--head_dim', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=None)
    parser.add_argument('--probes_path', type=str, required=True)
    parser.add_argument('--val_accs_path', type=str, required=True)
    parser.add_argument('--tuning_headwise_path', type=str, default='../features/llama2_chat_7B_nq_head_wise.npy')
    parser.add_argument('--scores_csv', type=str, required=True)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--results_root', type=str, default='./results_dump/llama-2-7b-instruct')
    # 试跑增强参数
    parser.add_argument('--include_strategies', type=str, default=None,
                        help='逗号分隔的策略名白名单，例如: layers_10_31,score_ge_0.7,topk_256_by_score,per_layer_top_4')
    parser.add_argument('--alphas', type=str, default=None,
                        help='逗号分隔的干预强度，例如: 2,6,10；不传则使用默认 2..20 步长2')
    parser.add_argument('--probe_factor_modes', type=str, default='both', choices=['both','true','false'],
                        help='是否乘探针分数：both/true/false')
    parser.add_argument('--limit_per_strategy', type=int, default=None,
                        help='每个策略最多选择的头数量；超过则按 CSV 分数降序截断')

    args = parser.parse_args()

    # 目录准备
    ans_dir = os.path.join(args.results_root, 'answer_dump')
    sum_dir = os.path.join(args.results_root, 'summary_dump')
    os.makedirs(ans_dir, exist_ok=True)
    os.makedirs(sum_dir, exist_ok=True)

    # 加载模型与分词器
    MODEL = HF_NAMES.get(args.model_name, None)
    if MODEL is None:
        raise ValueError(f"不支持的模型名: {args.model_name}")

    tokenizer = llama.LlamaTokenizerFast.from_pretrained(MODEL)
    dtype = torch.bfloat16 if 'llama3' in args.model_name else torch.float16
    model = llama.LlamaForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, torch_dtype=dtype, device_map='auto')
    device = model.device
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # 构造输入
    inputs, gold_answers_list = build_nq_generation_inputs(
        args.dataset_path,
        tokenizer,
        max_samples=None,
        max_docs=args.max_docs,
        use_chat_template=args.use_chat_template,
        sample_size=args.sample_size,
        sample_seed=args.sample_seed,
    )

    # 先跑一次 baseline
    baseline_ans, baseline_sum = run_baseline(model, tokenizer, device, inputs, gold_answers_list, args.max_new_tokens)
    baseline_sum.update({"alpha": 0.0, "model_name": args.model_name, "intervention": "none", "sample_size": args.sample_size})
    baseline_ans_path = os.path.join(ans_dir, 'nq_hparam_baseline_answers.jsonl')
    baseline_sum_path = os.path.join(sum_dir, 'nq_hparam_baseline_summary.json')
    with open(baseline_ans_path, 'w', encoding='utf-8') as f:
        for pred, golds in zip(baseline_ans, gold_answers_list):
            item = {"prediction": pred, "gold_answers": golds}
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    with open(baseline_sum_path, 'w', encoding='utf-8') as f:
        json.dump(baseline_sum, f, ensure_ascii=False, indent=2)

    # 即时打印：标准 RAG 指标与（相对基线）提升（为 0）
    print(
        f"[Baseline] samples={args.sample_size} EM={baseline_sum['EM']:.4f} "
        f"F1={baseline_sum['F1']:.4f} ΔEM={0.0:+.4f} ΔF1={0.0:+.4f}"
    )

    # 加载探针、分数与调强度激活
    with open(args.probes_path, 'rb') as f:
        probes = pickle.load(f)
    val_accs = np.load(args.val_accs_path)  # (L, H)
    tuning_headwise = np.load(args.tuning_headwise_path)  # (B, L, H*D)
    scores = load_scores_csv(args.scores_csv)
    L, H = infer_LH_from_scores(scores)
    selection_map = make_selection_strategies(scores, L, H, val_accs)
    # 策略白名单过滤
    if args.include_strategies:
        allow = {name.strip() for name in args.include_strategies.split(',') if name.strip()}
        selection_map = {k: v for k, v in selection_map.items() if k in allow}

    # 构建 (layer, head) -> score 映射，供限量与排序使用
    score_map: Dict[Tuple[int,int], float] = {(l,h): s for (l,h,s) in scores}

    # 超参数网格
    if args.alphas:
        try:
            alphas = [int(x) for x in args.alphas.split(',') if x.strip()]
        except Exception:
            raise ValueError('解析 --alphas 失败，请使用逗号分隔整数，例如 1,5,9')
    else:
        alphas = list(range(1, 10, 2))  # 1..9 步长2

    if args.probe_factor_modes == 'both':
        use_probe_factors = [False, True]
    elif args.probe_factor_modes == 'true':
        use_probe_factors = [True]
    else:
        use_probe_factors = [False]

    # 汇总表
    summary_rows = []

    for sel_name, sel_heads in selection_map.items():
        unique_heads = sorted(list({(l, h) for (l, h) in sel_heads}))
        # 按需限量：超过限制则按分数降序截断
        if args.limit_per_strategy and len(unique_heads) > args.limit_per_strategy:
            unique_heads = sorted(unique_heads, key=lambda lh: score_map.get(lh, 0.0), reverse=True)[:args.limit_per_strategy]
        k_count = len(unique_heads)
        if k_count == 0:
            continue

        for alpha in alphas:
            for use_pf in use_probe_factors:
                preds, summary = run_intervention(
                    model,
                    tokenizer,
                    device,
                    inputs,
                    gold_answers_list,
                    alpha,
                    sel_name,
                    unique_heads,
                    probes,
                    tuning_headwise,
                    args.head_dim,
                    args.num_heads,
                    use_pf,
                    val_accs,
                    args.max_new_tokens,
                )

                # 保存逐条答案
                ans_path = os.path.join(ans_dir, f"nq_hparam_{sel_name}_alpha{alpha}_pf{int(use_pf)}_answers.jsonl")
                with open(ans_path, 'w', encoding='utf-8') as f:
                    for pred, golds in zip(preds, gold_answers_list):
                        item = {"prediction": pred, "gold_answers": golds}
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')

                # 保存摘要
                out_summary = {
                    "EM": summary["EM"],
                    "F1": summary["F1"],
                    "alpha": float(alpha),
                    "model_name": args.model_name,
                    "intervention": sel_name,
                    "use_probe_factor": bool(use_pf),
                    "num_heads_selected": k_count,
                    "sample_size": args.sample_size,
                }
                sum_path = os.path.join(sum_dir, f"nq_hparam_{sel_name}_alpha{alpha}_pf{int(use_pf)}_summary.json")
                with open(sum_path, 'w', encoding='utf-8') as f:
                    json.dump(out_summary, f, ensure_ascii=False, indent=2)

                # 即时打印：当前实验的指标与相对基线的提升
                d_em = out_summary["EM"] - baseline_sum["EM"]
                d_f1 = out_summary["F1"] - baseline_sum["F1"]
                print(
                    (
                        f"[Intervene] strategy={sel_name} heads={k_count} alpha={alpha} pf={int(use_pf)} "
                        f"EM={out_summary['EM']:.4f} F1={out_summary['F1']:.4f} "
                        f"ΔEM={d_em:+.4f} ΔF1={d_f1:+.4f}"
                    )
                )

                # 记录汇总行
                summary_rows.append([
                    sel_name,
                    k_count,
                    alpha,
                    int(use_pf),
                    out_summary["EM"],
                    out_summary["F1"],
                ])

    # 保存最终汇总 CSV（按 F1 降序）
    final_csv = os.path.join(args.results_root, 'nq_hparam_search_summary.csv')
    summary_rows_sorted = sorted(summary_rows, key=lambda r: r[5], reverse=True)
    with open(final_csv, 'w', encoding='utf-8') as f:
        f.write('selection,num_heads,alpha,use_probe_factor,EM,F1\n')
        for row in summary_rows_sorted:
            f.write(','.join(map(str, row)) + '\n')

    print(f"Baseline summary saved to: {baseline_sum_path}")
    print(f"All run summaries saved under: {sum_dir}")
    print(f"Final hyperparam summary: {final_csv}")


if __name__ == '__main__':
    main()