import os
import argparse
import json
import csv
import pickle
import time
from typing import List, Tuple, Dict, Optional
import textwrap

import torch
import numpy as np
from tqdm import tqdm
from einops import rearrange

import llama
from baukit import TraceDict
from transformers import AutoTokenizer
from llama_utils import (
    _load_nq_jsonl,
    _build_messages_input,
    get_interventions_dict,
    evaluate_nq_em_f1,
    get_separated_activations_nq,
    get_com_directions,
)
from utils.prompts_templates import prompt_dict


# 模型路径映射（与现有脚本保持一致）
HF_NAMES = {
    'llama2_chat_7B': '/root/shared-nvme/RAG-llm/models/Llama-2-7b-chat-hf',
    'llama3_8B_instruct': '/root/shared-nvme/RAG-llm/models/Llama-3-8B-Instruct',
    'llama2_chat_13B': '/root/shared-nvme/RAG-llm/models/Llama-2-13b-chat-hf',
    'vicuna_7B_v1.5': '/root/shared-nvme/RAG-llm/models/vicuna-7b-v1.5',
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

        docs_block = "\n".join([f"Passage-{k+1}: {d}" for k, d in enumerate(docs_texts)])
        system_prompt = prompt_dict['qa']['RAG_system']
        user_prompt = prompt_dict['qa']['RAG_user'].format(paras=docs_block, question=question, answer='')

        input_ids = _build_messages_input(tokenizer, system_prompt, user_prompt, assistant_content=None, use_chat_template=use_chat_template)
        inputs.append(input_ids)
        gold_answers_list.append(list(answers))

    return inputs, gold_answers_list


def load_scores_csv(path: str) -> List[Tuple[int, int, float]]:
    items: List[Tuple[int, int, float]] = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            try:
                layer = int(row[0]); head = int(row[1]); score = float(row[2])
                items.append((layer, head, score))
            except Exception:
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
    saved_top_heads: Optional[List[Tuple[int,int]]] = None,
) -> Dict[str, List[Tuple[int, int]]]:
    by_score = sorted(scores, key=lambda x: x[2], reverse=True)
    strategies: Dict[str, List[Tuple[int, int]]] = {}

    # 分数阈值
    for thr in [0.5, 0.6, 0.7, 0.8, 0.9]:
        strategies[f'score_ge_{thr}'] = [(l, h) for (l, h, s) in scores if s >= thr]

    # 层区间（零基，右端 32 以包含第31层）
    def clamp_end(e: int) -> int:
        return min(e, L)

    strategies['layers_0_10'] = [(l, h) for (l, h, s) in scores if 0 <= l < clamp_end(10)]
    strategies['layers_10_20'] = [(l, h) for (l, h, s) in scores if 10 <= l < clamp_end(20)]
    strategies['layers_20_31'] = [(l, h) for (l, h, s) in scores if 20 <= l < clamp_end(32)]
    strategies['layers_10_31'] = [(l, h) for (l, h, s) in scores if 10 <= l < clamp_end(32)]
    strategies['layers_0_31'] = [(l, h) for (l, h, s) in scores if 0 <= l < clamp_end(32)]
    strategies['layers_0_5'] = [(l, h) for (l, h, s) in scores if 0 <= l < clamp_end(5)]
    strategies['layers_5_10'] = [(l, h) for (l, h, s) in scores if 5 <= l < clamp_end(10)]
    strategies['layers_5_9'] = [(l, h) for (l, h, s) in scores if 5 <= l < clamp_end(9)]
    strategies['layers_7_14'] = [(l, h) for (l, h, s) in scores if 7 <= l < clamp_end(14)]
    strategies['layers_8_15'] = [(l, h) for (l, h, s) in scores if 8 <= l < clamp_end(15)]
    strategies['layers_10_15'] = [(l, h) for (l, h, s) in scores if 10 <= l < clamp_end(15)]
    strategies['layers_15_20'] = [(l, h) for (l, h, s) in scores if 15 <= l < clamp_end(20)]
    strategies['layers_7_15'] = [(l, h) for (l, h, s) in scores if 7 <= l < clamp_end(15)]

    # Top Layers from Causal Trace Experiment
    strategies['top_k_layers_llama2_chat_7B'] = [(l, h) for (l, h, s) in scores if l in [11,2,13,15,16,12,10,8,6]]
    # strategies['top_k_layers_llama3_8B_instruct'] = [(l, h) for (l, h, s) in scores if l in [5, 30, 20, 2, 8, 11, 26, 0, 7, 22, 24, 21, 29]]
    strategies['top_k_layers_llama3_8B_instruct'] = [(l, h) for (l, h, s) in scores if l in [5, 30]]
    

    # 全部头
    strategies['all_heads'] = [(l, h) for (l, h, s) in scores]

    # 全局 top-k
    for k in [6, 8, 12, 13, 24, 32, 48, 64, 96, 128, 256, 448, 512, 768, 896, 1024]:
        kk = min(k, len(by_score))
        strategies[f'topk_{kk}_by_score'] = [(l, h) for (l, h, s) in by_score[:kk]]

    # 分层 top-m
    grouped: Dict[int, List[Tuple[int, int, float]]] = {l: [] for l in range(L)}
    for (l, h, s) in scores:
        grouped.setdefault(l, []).append((l, h, s))
    for m in [1, 2, 4, 8, 16]:
        sel: List[Tuple[int, int]] = []
        for l in range(L):
            heads_sorted = sorted(grouped.get(l, []), key=lambda x: x[2], reverse=True)
            sel.extend([(l, h) for (_, h, _) in heads_sorted[:m]])
        strategies[f'per_layer_top_{m}'] = sel

    # 读取已保存 top heads（可选）
    if saved_top_heads is not None:
        strategies['saved_top_heads'] = list(saved_top_heads)

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
    com_directions: Optional[np.ndarray] = None,
    pf_gamma: float = 1.0,
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
        use_center_of_mass=True,
        use_random_dir=False,
        com_directions=com_directions,
        probe_score_map=probe_score_map,
    )

    def lt_modulated_vector_add(head_output, layer_name, start_edit_location='lt'):
        h_out = rearrange(head_output, 'b s (h d) -> b s h d', h=num_heads_calc)
        # last token by default
        start_idx = -1 if start_edit_location == 'lt' else -1
        # 解析层索引
        try:
            layer_idx = int(str(layer_name).split('.')[2])
        except Exception:
            layer_idx = None
        for head, direction, proj_val_std, probe_factor in interventions[layer_name]:
            direction_to_add = torch.tensor(direction, dtype=h_out.dtype, device=h_out.device)
            alpha_cur = alpha
            if (alpha_per_layer is not None) and (layer_idx is not None) and (layer_idx in alpha_per_layer):
                alpha_cur = alpha_per_layer[layer_idx]
            proj_mult = proj_val_std
            reliability = float(probe_factor)
            strength_base = alpha_cur * proj_mult * (reliability ** pf_gamma)
            if start_idx == -1:
                # h_out[:, -1, head, :] += (strength_base * (1.0 - dynamic_score)).unsqueeze(-1) * direction_to_add
                h_out[:, -1, head, :] += strength_base * direction_to_add
            else:
                # h_out[:, start_idx:, head, :] += (strength_base * (1.0 - dynamic_score)).unsqueeze(-1) * direction_to_add
                h_out[:, start_idx:, head, :] += strength_base * direction_to_add
        return rearrange(h_out, 'b s h d -> b s (h d)')

    preds: List[str] = []
    layers_to_intervene = list(interventions.keys())
    start_time = time.time()
    timed_out = False
    with torch.no_grad():
        # 进度条描述：若是 alphamap，展示 tag
        desc = (f"intervene[{strategy_name}] alphamap pf={int(use_probe_factor)}" if alpha_per_layer is not None
                else f"intervene[{strategy_name}] alpha={alpha} pf={int(use_probe_factor)}")
        for input_ids in tqdm(inputs, desc=desc):
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

            if timeout_minutes is not None:
                elapsed_min = (time.time() - start_time) / 60.0
                if elapsed_min > timeout_minutes:
                    timed_out = True
                    break

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


def parse_range_or_list(text: Optional[str], default: List[float]) -> List[float]:
    if not text:
        return default
    t = text.strip()
    if t.startswith('range:'):
        try:
            _, a, b, s = t.split(':')
            a = float(a); b = float(b); s = float(s)
            vals = []
            v = a
            # 包含 b 边界（若步长整合到恰好达到），否则到达前一个
            while v <= b + 1e-8:
                vals.append(round(v, 6))
                v += s
            return vals
        except Exception:
            raise ValueError('range 格式错误，应为 range:start:stop:step')
    else:
        try:
            return [float(x) for x in t.split(',') if x.strip()]
        except Exception:
            raise ValueError('解析列表失败，请使用逗号分隔数值或 range:start:stop:step')


def main():
    parser = argparse.ArgumentParser(description='NQ 干预超参数搜索（整合版）')
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
    parser.add_argument('--tuning_headwise_path', type=str, default='./RAG/features/llama2_chat_7B_nq_head_wise.npy')
    parser.add_argument('--tuning_labels_path', type=str, default='./RAG/features/llama2_chat_7B_nq_labels.npy')
    parser.add_argument('--scores_csv', type=str, required=True)
    parser.add_argument('--saved_top_heads_path', type=str, default=None)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--results_root', type=str, default='./RAG/results_dump/llama-2-7b-instruct-unified')
    parser.add_argument('--pf_gamma', type=float, default=1.0)

    # 选择策略与过滤
    parser.add_argument('--include_strategies', type=str, default=None,
                        help='逗号分隔的策略白名单，例如: layers_10_31,score_ge_0.7,topk_256_by_score,per_layer_top_4,saved_top_heads')
    parser.add_argument('--limit_per_strategy', type=int, default=None,
                        help='每个策略最多选择的头数量；超过则按 CSV 分数降序截断')

    # 强度与乘因子
    parser.add_argument('--alphas', type=str, default='range:1:19:2',
                        help='干预强度列表或范围：逗号列表或 range:start:stop:step')
    parser.add_argument('--probe_factor_modes', type=str, default='both', choices=['both','true','false'],
                        help='是否乘探针分数：both/true/false')

    parser.add_argument('--timeout_minutes', type=float, default=4.0,
                        help='单次实验的最长运行时间（分钟），超过则中断并进入下一组参数')
    parser.add_argument('--skip_if_exists', action='store_true',
                        help='若当前实验的 summary 文件已存在则跳过执行')

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

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    dtype = torch.bfloat16 if 'llama3' in args.model_name else torch.float16
    model = llama.LlamaForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, torch_dtype=dtype, device_map='auto')
    device = model.device
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # 构造输入
    inputs, gold_answers_list = build_nq_generation_inputs(
        args.dataset_path, tokenizer, max_samples=None, max_docs=args.max_docs,
        use_chat_template=args.use_chat_template, sample_size=args.sample_size, sample_seed=args.sample_seed,
    )
##--------------------------------------------

    # 先跑一次 baseline
    baseline_ans, baseline_sum = run_baseline(model, tokenizer, device, inputs, gold_answers_list, args.max_new_tokens)
    baseline_sum.update({"alpha": 0.0, "model_name": args.model_name, "intervention": "none", "sample_size": args.sample_size})
    baseline_ans_path = os.path.join(ans_dir, 'nq_hparam_unified_baseline_answers.jsonl')
    baseline_sum_path = os.path.join(sum_dir, 'nq_hparam_unified_baseline_summary.json')
    with open(baseline_ans_path, 'w', encoding='utf-8') as f:
        for pred, golds in zip(baseline_ans, gold_answers_list):
            f.write(json.dumps({"prediction": pred, "gold_answers": golds}, ensure_ascii=False) + '\n')
    with open(baseline_sum_path, 'w', encoding='utf-8') as f:
        json.dump(baseline_sum, f, ensure_ascii=False, indent=2)

    print(
        f"[Baseline] samples={args.sample_size} EM={baseline_sum['EM']:.4f} "
        f"F1={baseline_sum['F1']:.4f} ΔEM={0.0:+.4f} ΔF1={0.0:+.4f}"
    )

##--------------------------------------------
    # 加载探针、分数与调强度激活
    with open(args.probes_path, 'rb') as f:
        probes = pickle.load(f)
    val_accs = np.load(args.val_accs_path)  # (L, H)
    tuning_headwise = np.load(args.tuning_headwise_path)  # (B, L, H*D)
    tuning_labels = np.load(args.tuning_labels_path)
    scores = load_scores_csv(args.scores_csv)
    L, H = infer_LH_from_scores(scores)

    # 计算质心均值偏移方向（复用 llama_utils）
    B_th, L_th, HD_th = tuning_headwise.shape
    if args.num_heads is None:
        if HD_th % args.head_dim != 0:
            raise ValueError('无法从特征维推断 num_heads，请显式传入 --num_heads')
        num_heads_calc = HD_th // args.head_dim
    else:
        num_heads_calc = args.num_heads
        
    tuning_sep = rearrange(tuning_headwise, 'b l (h d) -> b l h d', h=num_heads_calc, d=args.head_dim)
    num_questions = B_th // 2
    separated_head, separated_labels, _ = get_separated_activations_nq(tuning_labels, tuning_sep, num_questions)
    train_set_idxs = np.arange(num_questions)
    val_set_idxs = np.array([], dtype=int)
    com_directions = get_com_directions(L_th, num_heads_calc, train_set_idxs, val_set_idxs, separated_head, separated_labels)

    saved_top_heads = None
    if args.saved_top_heads_path:
        with open(args.saved_top_heads_path, 'rb') as f:
            obj = pickle.load(f)
            if isinstance(obj, dict):
                saved_top_heads = obj.get('top_heads', None) or obj.get('heads', None)
            elif isinstance(obj, list):
                saved_top_heads = obj

    selection_map = make_selection_strategies(scores, L, H, saved_top_heads=saved_top_heads)

    # 策略白名单过滤
    if args.include_strategies:
        allow = {name.strip() for name in args.include_strategies.split(',') if name.strip()}
        selection_map = {k: v for k, v in selection_map.items() if k in allow}

    # 构建 (layer, head) -> score 映射，供限量与排序使用
    score_map: Dict[Tuple[int,int], float] = {(l,h): s for (l,h,s) in scores}

    # 超参数网格
    alphas = parse_range_or_list(args.alphas, default=[float(x) for x in range(1, 20, 2)])
    if args.probe_factor_modes == 'both':
        use_probe_factors = [False, True]
    elif args.probe_factor_modes == 'true':
        use_probe_factors = [True]
    else:
        use_probe_factors = [False]

    summary_rows = []

    # 运行单策略网格
    for sel_name, sel_heads in selection_map.items():
        unique_heads = sorted(list({(l, h) for (l, h) in sel_heads}))
        if args.limit_per_strategy and len(unique_heads) > args.limit_per_strategy:
            unique_heads = sorted(unique_heads, key=lambda lh: score_map.get(lh, 0.0), reverse=True)[:args.limit_per_strategy]
        k_count = len(unique_heads)
        if k_count == 0:
            continue
        for alpha in alphas:
            for use_pf in use_probe_factors:
                sum_path = os.path.join(sum_dir, f"nq_unified_{sel_name}_alpha{alpha}_pf{int(use_pf)}_summary.json")
                if args.skip_if_exists and os.path.exists(sum_path):
                    print(f"[Skip] exists: {sum_path}")
                    continue
                preds, summary = run_intervention(
                    model, tokenizer, device, inputs, gold_answers_list,
                    float(alpha), sel_name, unique_heads, probes, tuning_headwise,
                    args.head_dim, args.num_heads, use_pf, 
                    val_accs, args.max_new_tokens, args.timeout_minutes, None,
                    com_directions=com_directions, pf_gamma=args.pf_gamma,
                )

                ans_path = os.path.join(ans_dir, f"nq_unified_{sel_name}_alpha{alpha}_pf{int(use_pf)}_answers.jsonl")
                with open(ans_path, 'w', encoding='utf-8') as f:
                    for pred, golds in zip(preds, gold_answers_list):
                        f.write(json.dumps({"prediction": pred, "gold_answers": golds}, ensure_ascii=False) + '\n')

                out_summary = {
                    "EM": summary["EM"], "F1": summary["F1"], "alpha": float(alpha),
                    "model_name": args.model_name, "intervention": sel_name,
                    "use_probe_factor": bool(use_pf), "num_heads_selected": k_count,
                    "sample_size": args.sample_size,
                    "timed_out": bool(summary.get("timed_out", False)),
                    "num_completed": int(summary.get("num_completed", len(preds))),
                    "elapsed_min": float(summary.get("elapsed_min", 0.0)),
                }
                with open(sum_path, 'w', encoding='utf-8') as f:
                    json.dump(out_summary, f, ensure_ascii=False, indent=2)
##--------------------------------------------

                d_em = out_summary["EM"] - baseline_sum["EM"]
                d_f1 = out_summary["F1"] - baseline_sum["F1"]
##--------------------------------------------
                status = "TIMED_OUT" if out_summary["timed_out"] else "OK"
                print((
                    f"[Intervene:{status}] strategy={sel_name} heads={k_count} alpha={alpha} pf={int(use_pf)} "
##--------------------------------------------
                    f"EM={out_summary['EM']:.4f} F1={out_summary['F1']:.4f} ΔEM={d_em:+.4f} ΔF1={d_f1:+.4f} "
##--------------------------------------------
                    f"completed={out_summary['num_completed']}/{args.sample_size} elapsed={out_summary['elapsed_min']:.2f}m"
                ))

                if not out_summary["timed_out"]:
                    summary_rows.append([sel_name, k_count, float(alpha), int(use_pf), out_summary["EM"], out_summary["F1"]])

    # 保存最终汇总 CSV（按 F1 降序）
    final_csv = os.path.join(args.results_root, 'nq_hparam_unified_summary.csv')
    summary_rows_sorted = sorted(summary_rows, key=lambda r: r[5], reverse=True)
    with open(final_csv, 'w', encoding='utf-8') as f:
        f.write('selection,num_heads,alpha/use_map,use_probe_factor,EM,F1\n')
        for row in summary_rows_sorted:
            f.write(','.join(map(str, row)) + '\n')
##--------------------------------------------
    print(f"Baseline summary saved to: {baseline_sum_path}")
##--------------------------------------------
    print(f"All run summaries saved under: {sum_dir}")
    print(f"Final hyperparam summary: {final_csv}")


if __name__ == '__main__':
    main()