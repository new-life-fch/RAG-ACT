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
from llama_utils import (
    _load_nq_jsonl,
    _build_messages_input,
    get_interventions_dict,
    evaluate_nq_em_f1,
    get_separated_activations_nq,
    get_com_directions,
)
from utils.prompts_templates import prompt_dict


# 模型路径映射
HF_NAMES = {
    'llama2_chat_7B': '/root/shared-nvme/RAG-llm/models/Llama-2-7b-chat-hf',
    'llama3_8B_instruct': '/root/shared-nvme/RAG-llm/models/Llama-3.1-8B-Instruct',
}

# 从 causal_layer_trace.csv 中读取的层排序 (Score = Delta_EM + Delta_F1 降序)
LAYER_ORDER = [
    8, 10, 7, 12, 13, 9, 11, 17, 5, 6, 
    22, 24, 30, 26, 21, 4, 29, 3, 25, 23, 
    16, 0, 19, 20, 18, 1, 2, 27, 28, 31, 
    15, 14
]

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
        system_prompt = prompt_dict['qa']['naive_RAG_system'].format(paras=docs_block)
        user_prompt = prompt_dict['qa']['naive_RAG_user'].format(question=question, answer='')

        input_ids = _build_messages_input(tokenizer, system_prompt, user_prompt, assistant_content=None, use_chat_template=use_chat_template)
        inputs.append(input_ids)
        gold_answers_list.append(list(answers))

    return inputs, gold_answers_list

def infer_LH_from_scores(scores: List[Tuple[int, int, float]]) -> Tuple[int, int]:
    L = max([l for l, _, _ in scores]) + 1 if scores else 0
    H = max([h for _, h, _ in scores]) + 1 if scores else 0
    return L, H

def make_selection_strategies(
    L: int,
    H: int,
) -> Dict[str, List[Tuple[int, int]]]:
    strategies: Dict[str, List[Tuple[int, int]]] = {}
    
    # 仅保留 Top-k 层干预策略
    # k 从 1 到 len(LAYER_ORDER)
    for k in range(1, len(LAYER_ORDER) + 1):
        top_layers = LAYER_ORDER[:k]
        # 选择这些层的所有头
        selected_heads = []
        for l in top_layers:
            for h in range(H):
                selected_heads.append((l, h))
        strategies[f'top_{k}_layers'] = selected_heads

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
            direction_to_add = torch.tensor(direction).to(h_out.device)
            alpha_cur = alpha
            proj_mult = proj_val_std
            reliability = float(probe_factor)
            try:
                if layer_idx is not None:
                    flat_idx = layer_idx * num_heads_calc + head
                    clf = probes[flat_idx]
                    w = torch.tensor(clf.coef_.reshape(-1), dtype=direction_to_add.dtype).to(h_out.device)
                    b = float(getattr(clf, 'intercept_', [0.0])[0])
                    x_vec = h_out[:, -1, head, :]
                    logit = (x_vec @ w) + b
                    dynamic_score = torch.sigmoid(logit)
                else:
                    dynamic_score = torch.tensor(0.0, dtype=direction_to_add.dtype, device=h_out.device)
            except Exception:
                dynamic_score = torch.tensor(0.0, dtype=direction_to_add.dtype, device=h_out.device)
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
        desc = f"intervene[{strategy_name}] alpha={alpha} pf={int(use_probe_factor)}"
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
    parser = argparse.ArgumentParser(description='NQ Top-k Layer Intervention Search')
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
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--results_root', type=str, default='./RAG/results_dump/llama-2-7b-instruct-top-layers')
    parser.add_argument('--pf_gamma', type=float, default=1.0)
    
    # 移除 strategies 参数，直接内置 top-k 策略
    
    # 强度与乘因子
    parser.add_argument('--alphas', type=str, default='range:1:19:2',
                        help='干预强度列表或范围：逗号列表或 range:start:stop:step')
    parser.add_argument('--probe_factor_modes', type=str, default='both', choices=['both','true','false'],
                        help='是否乘探针分数：both/true/false')

    # 移除 composites 相关参数

    parser.add_argument('--timeout_minutes', type=float, default=15.0,
                        help='单次实验的最长运行时间（分钟），超过则中断并进入下一组参数')
    parser.add_argument('--skip_if_exists', action='store_true',
                        help='若当前实验的 summary 文件已存在则跳过执行')
    
    # 汇总 CSV 输出路径
    parser.add_argument('--summary_csv', type=str, default='top_layer_intervention_results.csv',
                        help='汇总所有实验结果的 CSV 文件名')

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
        args.dataset_path, tokenizer, max_samples=None, max_docs=args.max_docs,
        use_chat_template=args.use_chat_template, sample_size=args.sample_size, sample_seed=args.sample_seed,
    )

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
    
    baseline_em = baseline_sum['EM']
    baseline_f1 = baseline_sum['F1']

    # 加载探针、分数与调强度激活
    with open(args.probes_path, 'rb') as f:
        probes = pickle.load(f)
    val_accs = np.load(args.val_accs_path)  # (L, H)
    tuning_headwise = np.load(args.tuning_headwise_path)  # (B, L, H*D)
    tuning_labels = np.load(args.tuning_labels_path)
    # 不再需要 scores_csv，因为层顺序已硬编码

    L = val_accs.shape[0]
    H = val_accs.shape[1]

    # 计算质心均值偏移方向
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

    # 生成 Top-k 策略
    selection_map = make_selection_strategies(L, H)

    # 解析参数列表
    alpha_list = parse_range_or_list(args.alphas, [1.0])
    pf_modes = []
    if args.probe_factor_modes in ['both', 'true']:
        pf_modes.append(True)
    if args.probe_factor_modes in ['both', 'false']:
        pf_modes.append(False)

    # 汇总数据列表
    summary_rows = []
    # 添加 baseline
    summary_rows.append({
        'strategy': 'baseline',
        'k': 0,
        'alpha': 0.0,
        'pf': False,
        'EM': baseline_em,
        'F1': baseline_f1,
        'Delta_EM': 0.0,
        'Delta_F1': 0.0
    })

    # 遍历执行
    for strat_name, heads_list in selection_map.items():
        # 解析 k 值
        try:
            k = int(strat_name.split('_')[1])
        except:
            k = -1 # Should not happen
            
        for use_pf in pf_modes:
            for alpha in alpha_list:
                run_name = f"{strat_name}_alpha_{alpha}_pf_{int(use_pf)}"
                
                # 检查是否存在
                sum_path = os.path.join(sum_dir, f"{run_name}_summary.json")
                if args.skip_if_exists and os.path.exists(sum_path):
                    print(f"Skipping {run_name} (exists)")
                    # 尝试读取已存在的结果加入汇总
                    try:
                        with open(sum_path, 'r') as f:
                            res = json.load(f)
                        summary_rows.append({
                            'strategy': strat_name,
                            'k': k,
                            'alpha': alpha,
                            'pf': use_pf,
                            'EM': res['EM'],
                            'F1': res['F1'],
                            'Delta_EM': res['EM'] - baseline_em,
                            'Delta_F1': res['F1'] - baseline_f1
                        })
                    except:
                        pass
                    continue

                print(f"Running {run_name} (heads={len(heads_list)})...")
                preds, metrics = run_intervention(
                    model, tokenizer, device, inputs, gold_answers_list,
                    alpha=alpha,
                    strategy_name=strat_name,
                    top_heads=heads_list,
                    probes=probes,
                    tuning_headwise=tuning_headwise,
                    head_dim=args.head_dim,
                    num_heads=num_heads_calc,
                    use_probe_factor=use_pf,
                    val_accs=val_accs,
                    max_new_tokens=args.max_new_tokens,
                    timeout_minutes=args.timeout_minutes,
                    com_directions=com_directions,
                    pf_gamma=args.pf_gamma,
                )

                # 保存结果
                metrics.update({
                    "alpha": alpha,
                    "strategy": strat_name,
                    "use_probe_factor": use_pf,
                    "num_heads": len(heads_list),
                    "pf_gamma": args.pf_gamma
                })
                
                ans_path = os.path.join(ans_dir, f"{run_name}_answers.jsonl")
                with open(ans_path, 'w', encoding='utf-8') as f:
                    for pred, golds in zip(preds, gold_answers_list):
                        f.write(json.dumps({"prediction": pred, "gold_answers": golds}, ensure_ascii=False) + '\n')
                with open(sum_path, 'w', encoding='utf-8') as f:
                    json.dump(metrics, f, ensure_ascii=False, indent=2)

                delta_em = metrics['EM'] - baseline_em
                delta_f1 = metrics['F1'] - baseline_f1
                
                print(f"  -> EM={metrics['EM']:.4f} (Δ={delta_em:+.4f}), F1={metrics['F1']:.4f} (Δ={delta_f1:+.4f})")
                
                summary_rows.append({
                    'strategy': strat_name,
                    'k': k,
                    'alpha': alpha,
                    'pf': use_pf,
                    'EM': metrics['EM'],
                    'F1': metrics['F1'],
                    'Delta_EM': delta_em,
                    'Delta_F1': delta_f1
                })

    # 保存汇总 CSV
    csv_path = os.path.join(args.results_root, args.summary_csv)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    print(f"Saving summary to {csv_path}...")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['strategy', 'k', 'alpha', 'pf', 'EM', 'F1', 'Delta_EM', 'Delta_F1'])
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

if __name__ == "__main__":
    main()
