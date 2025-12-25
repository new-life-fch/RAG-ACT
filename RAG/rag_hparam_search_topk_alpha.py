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
    _load_data_jsonl,
    _build_messages_input,
    get_interventions_dict,
    evaluate_rag_em_f1,
    get_separated_activations_dataset,
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
    entries = _load_data_jsonl(jsonl_path, max_samples=max_samples)

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
    em_b, f1_b = evaluate_rag_em_f1(preds_baseline, gold_answers_list)
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
            strength_base = alpha_cur * proj_mult * reliability
            if start_idx == -1:
                h_out[:, -1, head, :] += strength_base * direction_to_add
            else:
                h_out[:, start_idx:, head, :] += strength_base * direction_to_add
        return rearrange(h_out, 'b s h d -> b s (h d)')

    preds: List[str] = []
    layers_to_intervene = list(interventions.keys())
    start_time = time.time()
    timed_out = False
    with torch.no_grad():
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
        em_i, f1_i = evaluate_rag_em_f1(preds, subset_golds) if len(preds) else (0.0, 0.0)
        return preds, {
            "EM": float(em_i),
            "F1": float(f1_i),
            "timed_out": True,
            "num_completed": len(preds),
            "elapsed_min": (time.time() - start_time) / 60.0,
        }
    else:
        em_i, f1_i = evaluate_rag_em_f1(preds, gold_answers_list)
        return preds, {
            "EM": float(em_i),
            "F1": float(f1_i),
            "timed_out": False,
            "num_completed": len(preds),
            "elapsed_min": (time.time() - start_time) / 60.0,
        }


def main():
    parser = argparse.ArgumentParser(description='NQ 干预超参数搜索（仅 Top-K 和 Alpha）')
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
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--results_root', type=str, default='./RAG/results/llama-2-7b-instruct-topk-alpha')
    parser.add_argument('--timeout_minutes', type=float, default=10)
    parser.add_argument('--skip_if_exists', action='store_true')
    
    args = parser.parse_args()

    # ---------------------------------------------------------
    # 1. 定义超参数网格 (Reasonable Grids)
    # ---------------------------------------------------------
    # Top-K: 从小到大，覆盖主要区间
    # top_k_grid = [1, 19, 30, 39, 50, 62, 71, 87, 100, 120, 136] #vicuna
    # top_k_grid = [2, 15, 28, 35, 43, 57, 69, 92, 106, 131, 154] #llama2
    top_k_grid = [5,12,20,32,47,60,84,112,133,154,173] #llama3

    
    # Alpha: 覆盖弱到强 (3 ~ 9)
    alpha_grid = [3,5,7,9]

    
    # Probe Factor Mode: 为了控制变量，这里固定为 False (不乘探针准确率) 
    # 如果要对比，可以设为 [False] 或 [True]。这里默认设为 False (纯强度控制)
    # 这里我们选择 False (不乘 Probe Factor)，让 Alpha 成为唯一的缩放因子。
    use_probe_factors = [False] 

    # ---------------------------------------------------------
    # 2. 准备目录
    # ---------------------------------------------------------
    ans_dir = os.path.join(args.results_root, 'answer_dump')
    sum_dir = os.path.join(args.results_root, 'summary_dump')
    os.makedirs(ans_dir, exist_ok=True)
    os.makedirs(sum_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 3. 加载模型
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # 4. 构造数据
    # ---------------------------------------------------------
    inputs, gold_answers_list = build_nq_generation_inputs(
        args.dataset_path, tokenizer, max_samples=None, max_docs=args.max_docs,
        use_chat_template=args.use_chat_template, sample_size=args.sample_size, sample_seed=args.sample_seed,
    )

    # ---------------------------------------------------------
    # 5. Baseline
    # ---------------------------------------------------------
    baseline_ans, baseline_sum = run_baseline(model, tokenizer, device, inputs, gold_answers_list, args.max_new_tokens)
    baseline_sum.update({"alpha": 0.0, "model_name": args.model_name, "intervention": "none", "sample_size": args.sample_size})
    baseline_ans_path = os.path.join(ans_dir, 'baseline_answers.jsonl')
    baseline_sum_path = os.path.join(sum_dir, 'baseline_summary.json')
    with open(baseline_ans_path, 'w', encoding='utf-8') as f:
        for pred, golds in zip(baseline_ans, gold_answers_list):
            f.write(json.dumps({"prediction": pred, "gold_answers": golds}, ensure_ascii=False) + '\n')
    with open(baseline_sum_path, 'w', encoding='utf-8') as f:
        json.dump(baseline_sum, f, ensure_ascii=False, indent=2)

    print(f"[Baseline] EM={baseline_sum['EM']:.4f} F1={baseline_sum['F1']:.4f}")

    # ---------------------------------------------------------
    # 6. 加载资源 (Probes, Features, Scores)
    # ---------------------------------------------------------
    with open(args.probes_path, 'rb') as f:
        probes = pickle.load(f)
    val_accs = np.load(args.val_accs_path)
    tuning_headwise = np.load(args.tuning_headwise_path)
    tuning_labels = np.load(args.tuning_labels_path)
    scores = load_scores_csv(args.scores_csv)
    L, H = infer_LH_from_scores(scores)
    by_score = sorted(scores, key=lambda x: x[2], reverse=True) # 按分数降序

    # 计算质心方向
    B_th, L_th, HD_th = tuning_headwise.shape
    if args.num_heads is None:
        num_heads_calc = HD_th // args.head_dim
    else:
        num_heads_calc = args.num_heads
    
    tuning_sep = rearrange(tuning_headwise, 'b l (h d) -> b l h d', h=num_heads_calc, d=args.head_dim)
    num_questions = B_th // 2
    separated_head, separated_labels, _ = get_separated_activations_dataset(tuning_labels, tuning_sep, num_questions)
    train_set_idxs = np.arange(num_questions)
    val_set_idxs = np.array([], dtype=int)
    com_directions = get_com_directions(L_th, num_heads_calc, train_set_idxs, val_set_idxs, separated_head, separated_labels)

    # ---------------------------------------------------------
    # 7. 主循环 (Top-K x Alpha)
    # ---------------------------------------------------------
    summary_rows = []
    
    # 遍历 Top-K
    for k in top_k_grid:
        kk = min(k, len(by_score))
        sel_name = f'topk_{kk}_by_score'
        sel_heads = [(l, h) for (l, h, s) in by_score[:kk]]
        
        # 遍历 Alpha
        for alpha in alpha_grid:
            for use_pf in use_probe_factors:
                sum_path = os.path.join(sum_dir, f"{sel_name}_alpha{alpha}_pf{int(use_pf)}_summary.json")
                if args.skip_if_exists and os.path.exists(sum_path):
                    print(f"[Skip] {sum_path}")
                    continue
                
                preds, summary = run_intervention(
                    model, tokenizer, device, inputs, gold_answers_list,
                    float(alpha), sel_name, sel_heads, probes, tuning_headwise,
                    args.head_dim, args.num_heads, use_pf, 
                    val_accs, args.max_new_tokens, args.timeout_minutes, None,
                    com_directions=com_directions,
                )

                ans_path = os.path.join(ans_dir, f"{sel_name}_alpha{alpha}_pf{int(use_pf)}_answers.jsonl")
                with open(ans_path, 'w', encoding='utf-8') as f:
                    for pred, golds in zip(preds, gold_answers_list):
                        f.write(json.dumps({"prediction": pred, "gold_answers": golds}, ensure_ascii=False) + '\n')
                
                out_summary = {
                    "EM": summary["EM"], "F1": summary["F1"], "alpha": float(alpha),
                    "model_name": args.model_name, "intervention": sel_name,
                    "use_probe_factor": bool(use_pf), "num_heads_selected": kk,
                    "sample_size": args.sample_size,
                    "timed_out": bool(summary.get("timed_out", False)),
                }
                with open(sum_path, 'w', encoding='utf-8') as f:
                    json.dump(out_summary, f, ensure_ascii=False, indent=2)

                d_em = out_summary["EM"] - baseline_sum["EM"]
                d_f1 = out_summary["F1"] - baseline_sum["F1"]
                print(f"[Run] {sel_name} alpha={alpha} pf={int(use_pf)} -> EM={out_summary['EM']:.4f} F1={out_summary['F1']:.4f} d_EM={d_em:.4f} d_F1={d_f1:.4f}")
                
                if not out_summary["timed_out"]:
                    summary_rows.append([sel_name, kk, float(alpha), int(use_pf), out_summary["EM"], out_summary["F1"], d_em, d_f1])

    # ---------------------------------------------------------
    # 8. 保存汇总
    # ---------------------------------------------------------
    final_csv = os.path.join(args.results_root, 'final_summary.csv')
    summary_rows_sorted = sorted(summary_rows, key=lambda r: r[5], reverse=True)
    with open(final_csv, 'w', encoding='utf-8') as f:
        f.write('selection,num_heads,alpha,use_probe_factor,EM,F1,d_EM,d_F1\n')
        for row in summary_rows_sorted:
            f.write(','.join(map(str, row)) + '\n')
            
    print(f"Done. Results saved to {args.results_root}")

if __name__ == '__main__':
    main()
