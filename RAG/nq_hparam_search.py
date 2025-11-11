import os
import argparse
import json
import csv
import pickle
import time
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
    # compute_em_f1  # 若需要逐样本计算可启用
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
    timeout_minutes: Optional[float] = None,
    alpha_per_layer: Optional[Dict[int, float]] = None,
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
        # 解析层索引（model.layers.{i}.self_attn.head_out）
        try:
            layer_idx = int(str(layer_name).split('.')[2])
        except Exception:
            layer_idx = None
        for head, direction, proj_val_std, probe_factor in interventions[layer_name]:
            direction_to_add = torch.tensor(direction).to(h_out.device)
            # 按层选择强度：若提供 alpha_per_layer，则优先使用该层的强度
            alpha_cur = alpha
            if (alpha_per_layer is not None) and (layer_idx is not None) and (layer_idx in alpha_per_layer):
                alpha_cur = alpha_per_layer[layer_idx]
            strength = alpha_cur * proj_val_std * probe_factor
            if start_idx == -1:
                h_out[:, -1, head, :] += strength * direction_to_add
            else:
                h_out[:, start_idx:, head, :] += strength * direction_to_add
        return rearrange(h_out, 'b s h d -> b s (h d)')

    preds: List[str] = []
    layers_to_intervene = list(interventions.keys())
    start_time = time.time()
    timed_out = False
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

            # 检测单次实验超时并中断
            if timeout_minutes is not None:
                elapsed_min = (time.time() - start_time) / 60.0
                if elapsed_min > timeout_minutes:
                    timed_out = True
                    break

    # 计算指标：若超时，则仅对已完成样本计算平均
    if timed_out:
        subset_golds = gold_answers_list[:len(preds)]
        em_i, f1_i = evaluate_nq_em_f1(preds, subset_golds) if len(preds) else (0.0, 0.0)
        return preds, {
            "EM": float(em_i),
            "F1": float(f1_i),
            "timed_out": True,
            "num_completed": len(preds),
            "elapsed_min": (time.time() - start_time) / 60.0,
        }
    else:
        em_i, f1_i = evaluate_nq_em_f1(preds, gold_answers_list)
        return preds, {
            "EM": float(em_i),
            "F1": float(f1_i),
            "timed_out": False,
            "num_completed": len(preds),
            "elapsed_min": (time.time() - start_time) / 60.0,
        }


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
                        help='逗号分隔的干预强度（支持浮点数），例如: 2,6,10 或 0.7；不传则使用默认 1..19 步长2')
    parser.add_argument('--probe_factor_modes', type=str, default='both', choices=['both','true','false'],
                        help='是否乘探针分数：both/true/false')
    parser.add_argument('--limit_per_strategy', type=int, default=None,
                        help='每个策略最多选择的头数量；超过则按 CSV 分数降序截断')
    parser.add_argument('--supplement', action='store_true',
                        help='启用补充实验预设参数组合，并将结果保存到 results_dump/llama-2-7b-instruct-supplement')
    parser.add_argument('--timeout_minutes', type=float, default=6.0,
                        help='单次实验的最长运行时间（分钟），超过则中断并进入下一组参数')

    args = parser.parse_args()

    # 补充模式：切换结果目录到 supplement
    if args.supplement:
        args.results_root = './results_dump/llama-2-7b-instruct-supplement'

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

    # 超参数网格（补充模式下先执行复合方案，再执行各策略的单独强度）
    if args.alphas:
        try:
            alphas = [float(x) for x in args.alphas.split(',') if x.strip()]
        except Exception:
            raise ValueError('解析 --alphas 失败，请使用逗号分隔数字，例如 1,5,9 或 0.7')
    else:
        alphas = [float(x) for x in range(1, 20, 2)]  # 1..19 步长2

    if args.supplement:
        use_probe_factors = [False, True]
    else:
        if args.probe_factor_modes == 'both':
            use_probe_factors = [False, True]
        elif args.probe_factor_modes == 'true':
            use_probe_factors = [True]
        else:
            use_probe_factors = [False]

    # 汇总表
    summary_rows = []

    # 补充实验的预设参数组合：策略 -> 专属 alphas
    supplement_plan: Dict[str, List[float]] = {
        'layers_0_10': [5.0],
        'layers_10_31': [7.0],
        'layers_20_31': [11.0, 13.0, 15.0],  # 含单独 9.0 以跑第二点
        'layers_10_20': [10.0, 11.0, 12.0, 13.0, 2.0, 0.7, 0.5, 0.3],
    }

    # 先执行补充模式下的复合方案（不同层区间同时不同强度）
    if args.supplement:
        # 复合 1：0-9→5，10-31→7
        if ('layers_0_10' in selection_map) and ('layers_10_31' in selection_map):
            comp_name = 'layers_0_10+layers_10_31'
            comp_heads = list(set(selection_map['layers_0_10']) | set(selection_map['layers_10_31']))
            alpha_map = {l: 5.0 for l in range(0, 10)}
            alpha_map.update({l: 7.0 for l in range(10, min(32, L))})
            unique_heads = sorted(list({(l, h) for (l, h) in comp_heads}))
            if args.limit_per_strategy and len(unique_heads) > args.limit_per_strategy:
                unique_heads = sorted(unique_heads, key=lambda lh: score_map.get(lh, 0.0), reverse=True)[:args.limit_per_strategy]
            k_count = len(unique_heads)
            if k_count:
                for use_pf in use_probe_factors:
                    preds, summary = run_intervention(
                        model, tokenizer, device, inputs, gold_answers_list,
                        0.0, comp_name, unique_heads, probes, tuning_headwise,
                        args.head_dim, args.num_heads, use_pf, val_accs,
                        args.max_new_tokens, args.timeout_minutes, alpha_map,
                    )
                    ans_path = os.path.join(ans_dir, f"nq_hparam_{comp_name}_alphamap_0-9=5_10-31=7_pf{int(use_pf)}_answers.jsonl")
                    with open(ans_path, 'w', encoding='utf-8') as f:
                        for pred, golds in zip(preds, gold_answers_list):
                            f.write(json.dumps({"prediction": pred, "gold_answers": golds}, ensure_ascii=False) + '\n')
                    out_summary = {
                        "EM": summary["EM"], "F1": summary["F1"], "alpha": "map",
                        "alpha_map": {"0-9": 5.0, "10-31": 7.0},
                        "model_name": args.model_name, "intervention": comp_name,
                        "use_probe_factor": bool(use_pf), "num_heads_selected": k_count,
                        "sample_size": args.sample_size,
                        "timed_out": bool(summary.get("timed_out", False)),
                        "num_completed": int(summary.get("num_completed", len(preds))),
                        "elapsed_min": float(summary.get("elapsed_min", 0.0)),
                    }
                    sum_path = os.path.join(sum_dir, f"nq_hparam_{comp_name}_alphamap_0-9=5_10-31=7_pf{int(use_pf)}_summary.json")
                    with open(sum_path, 'w', encoding='utf-8') as f:
                        json.dump(out_summary, f, ensure_ascii=False, indent=2)
                    d_em = out_summary["EM"] - baseline_sum["EM"]
                    d_f1 = out_summary["F1"] - baseline_sum["F1"]
                    status = "TIMED_OUT" if out_summary["timed_out"] else "OK"
                    print((
                        f"[Intervene:{status}] strategy={comp_name} heads={k_count} alphamap=0-9=5_10-31=7 pf={int(use_pf)} "
                        f"EM={out_summary['EM']:.4f} F1={out_summary['F1']:.4f} ΔEM={d_em:+.4f} ΔF1={d_f1:+.4f} "
                        f"completed={out_summary['num_completed']}/{args.sample_size} elapsed={out_summary['elapsed_min']:.2f}m"
                    ))
                    if not out_summary["timed_out"]:
                        summary_rows.append([comp_name, k_count, 'map_0-9=5_10-31=7', int(use_pf), out_summary["EM"], out_summary["F1"]])

        # 复合 2：0-9→5，20-31→9
        if ('layers_0_10' in selection_map) and ('layers_20_31' in selection_map):
            comp_name = 'layers_0_10+layers_20_31'
            comp_heads = list(set(selection_map['layers_0_10']) | set(selection_map['layers_20_31']))
            alpha_map = {l: 5.0 for l in range(0, 10)}
            alpha_map.update({l: 9.0 for l in range(20, min(32, L))})
            unique_heads = sorted(list({(l, h) for (l, h) in comp_heads}))
            if args.limit_per_strategy and len(unique_heads) > args.limit_per_strategy:
                unique_heads = sorted(unique_heads, key=lambda lh: score_map.get(lh, 0.0), reverse=True)[:args.limit_per_strategy]
            k_count = len(unique_heads)
            if k_count:
                for use_pf in use_probe_factors:
                    preds, summary = run_intervention(
                        model, tokenizer, device, inputs, gold_answers_list,
                        0.0, comp_name, unique_heads, probes, tuning_headwise,
                        args.head_dim, args.num_heads, use_pf, val_accs,
                        args.max_new_tokens, args.timeout_minutes, alpha_map,
                    )
                    ans_path = os.path.join(ans_dir, f"nq_hparam_{comp_name}_alphamap_0-9=5_20-31=9_pf{int(use_pf)}_answers.jsonl")
                    with open(ans_path, 'w', encoding='utf-8') as f:
                        for pred, golds in zip(preds, gold_answers_list):
                            f.write(json.dumps({"prediction": pred, "gold_answers": golds}, ensure_ascii=False) + '\n')
                    out_summary = {
                        "EM": summary["EM"], "F1": summary["F1"], "alpha": "map",
                        "alpha_map": {"0-9": 5.0, "20-31": 9.0},
                        "model_name": args.model_name, "intervention": comp_name,
                        "use_probe_factor": bool(use_pf), "num_heads_selected": k_count,
                        "sample_size": args.sample_size,
                        "timed_out": bool(summary.get("timed_out", False)),
                        "num_completed": int(summary.get("num_completed", len(preds))),
                        "elapsed_min": float(summary.get("elapsed_min", 0.0)),
                    }
                    sum_path = os.path.join(sum_dir, f"nq_hparam_{comp_name}_alphamap_0-9=5_20-31=9_pf{int(use_pf)}_summary.json")
                    with open(sum_path, 'w', encoding='utf-8') as f:
                        json.dump(out_summary, f, ensure_ascii=False, indent=2)
                    d_em = out_summary["EM"] - baseline_sum["EM"]
                    d_f1 = out_summary["F1"] - baseline_sum["F1"]
                    status = "TIMED_OUT" if out_summary["timed_out"] else "OK"
                    print((
                        f"[Intervene:{status}] strategy={comp_name} heads={k_count} alphamap=0-9=5_20-31=9 pf={int(use_pf)} "
                        f"EM={out_summary['EM']:.4f} F1={out_summary['F1']:.4f} ΔEM={d_em:+.4f} ΔF1={d_f1:+.4f} "
                        f"completed={out_summary['num_completed']}/{args.sample_size} elapsed={out_summary['elapsed_min']:.2f}m"
                    ))
                    if not out_summary["timed_out"]:
                        summary_rows.append([comp_name, k_count, 'map_0-9=5_20-31=9', int(use_pf), out_summary["EM"], out_summary["F1"]])

    # 常规/补充的单策略循环
    for sel_name, sel_heads in selection_map.items():
        # 补充模式：仅保留设定的策略
        if args.supplement and sel_name not in supplement_plan:
            continue
        # 补充模式：仅保留设定的策略
        if args.supplement and sel_name not in supplement_plan:
            continue
        unique_heads = sorted(list({(l, h) for (l, h) in sel_heads}))
        # 按需限量：超过限制则按分数降序截断
        if args.limit_per_strategy and len(unique_heads) > args.limit_per_strategy:
            unique_heads = sorted(unique_heads, key=lambda lh: score_map.get(lh, 0.0), reverse=True)[:args.limit_per_strategy]
        k_count = len(unique_heads)
        if k_count == 0:
            continue

        # 针对补充模式，使用每个策略的专属 alphas；否则使用通用 alphas
        local_alphas = supplement_plan[sel_name] if args.supplement else alphas
        for alpha in local_alphas:
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
                    args.timeout_minutes,
                    None,
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
                    "timed_out": bool(summary.get("timed_out", False)),
                    "num_completed": int(summary.get("num_completed", len(preds))),
                    "elapsed_min": float(summary.get("elapsed_min", 0.0)),
                }
                sum_path = os.path.join(sum_dir, f"nq_hparam_{sel_name}_alpha{alpha}_pf{int(use_pf)}_summary.json")
                with open(sum_path, 'w', encoding='utf-8') as f:
                    json.dump(out_summary, f, ensure_ascii=False, indent=2)

                # 即时打印：当前实验的指标与相对基线的提升与耗时
                d_em = out_summary["EM"] - baseline_sum["EM"]
                d_f1 = out_summary["F1"] - baseline_sum["F1"]
                status = "TIMED_OUT" if out_summary["timed_out"] else "OK"
                print(
                    (
                        f"[Intervene:{status}] strategy={sel_name} heads={k_count} alpha={alpha} pf={int(use_pf)} "
                        f"EM={out_summary['EM']:.4f} F1={out_summary['F1']:.4f} "
                        f"ΔEM={d_em:+.4f} ΔF1={d_f1:+.4f} "
                        f"completed={out_summary['num_completed']}/{args.sample_size} elapsed={out_summary['elapsed_min']:.2f}m"
                    )
                )

                # 记录汇总行（仅收录未超时实验，避免不公平排序）
                if not out_summary["timed_out"]:
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