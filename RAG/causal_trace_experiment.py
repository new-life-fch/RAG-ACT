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
from transformers import AutoTokenizer
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
            continue

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

def run_layer_intervention(
    model,
    tokenizer,
    device,
    inputs: List[torch.Tensor],
    gold_answers_list: List[List[str]],
    layer_idx: int,
    alpha: float,
    probes: List,
    tuning_headwise: np.ndarray,
    head_dim: int,
    num_heads: int,
    val_accs: Optional[np.ndarray],
    max_new_tokens: int,
    com_directions: Optional[np.ndarray] = None,
) -> Tuple[List[str], Dict[str, float]]:
    
    # 构造当前层的干预字典：干预该层的所有头
    top_heads = [(layer_idx, h) for h in range(num_heads)]
    
    B, L, HD = tuning_headwise.shape
    tuning_sep = rearrange(tuning_headwise, 'b l (h d) -> b l h d', h=num_heads, d=head_dim)
    
    # 获取干预参数
    # 这里我们使用 get_interventions_dict 来获取每个头的方向和投影标准差
    # 注意：我们这里简化逻辑，始终使用 True Truth 的方向（即 use_center_of_mass=True）
    interventions = get_interventions_dict(
        top_heads,
        probes,
        tuning_sep,
        num_heads,
        use_center_of_mass=True,
        use_random_dir=False,
        com_directions=com_directions,
        probe_score_map=val_accs, # 使用 val_accs 作为 probe_factor
    )

    def lt_modulated_vector_add(head_output, layer_name, start_edit_location='lt'):
        h_out = rearrange(head_output, 'b s (h d) -> b s h d', h=num_heads)
        
        # 解析层名确认是否匹配（防御性编程）
        try:
            current_layer_idx = int(str(layer_name).split('.')[2])
        except Exception:
            current_layer_idx = None
            
        if current_layer_idx != layer_idx:
             return head_output # Should not happen if TraceDict is set up correctly

        for head, direction, proj_val_std, probe_factor in interventions[layer_name]:
            direction_to_add = torch.tensor(direction, dtype=h_out.dtype, device=h_out.device)
            
            # 计算动态强度
            reliability = float(probe_factor) if probe_factor is not None else 1.0
            # strength = alpha * proj_val_std * reliability
            strength = alpha * proj_val_std * 1
            
            h_out[:, -1, head, :] += strength * direction_to_add

        return rearrange(h_out, 'b s h d -> b s (h d)')

    preds: List[str] = []
    layers_to_intervene = list(interventions.keys())
    
    with torch.no_grad():
        for input_ids in inputs: # 简化版，不显示每条的进度条，只显示层的进度
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

    em, f1 = evaluate_nq_em_f1(preds, gold_answers_list)
    return preds, {"EM": em, "F1": f1}


def main():
    parser = argparse.ArgumentParser(description='NQ Causal Trace Simplified Experiment')
    parser.add_argument('--model_name', type=str, default='llama2_chat_7B')
    parser.add_argument('--dataset_path', type=str, default='/root/shared-nvme/RAG-llm/RAG/data/test.jsonl')
    parser.add_argument('--use_chat_template', action='store_true')
    parser.add_argument('--max_docs', type=int, default=5)
    parser.add_argument('--sample_size', type=int, default=100) # 默认跑100条，速度快一点
    parser.add_argument('--sample_seed', type=int, default=2025)
    parser.add_argument('--head_dim', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=32) # Llama 2 7B 默认 32 头
    parser.add_argument('--num_layers', type=int, default=32) # Llama 2 7B 默认 32 层
    
    # 必须的特征文件
    parser.add_argument('--probes_path', type=str, required=True)
    parser.add_argument('--val_accs_path', type=str, required=True)
    parser.add_argument('--tuning_headwise_path', type=str, default='./RAG/features/llama2_chat_7B_nq_head_wise.npy')
    parser.add_argument('--tuning_labels_path', type=str, default='./RAG/features/llama2_chat_7B_nq_labels.npy')
    
    parser.add_argument('--max_new_tokens', type=int, default=64)
    parser.add_argument('--output_csv', type=str, default='causal_trace_results.csv')
    parser.add_argument('--alpha', type=float, default=15.0, help="Intervention strength alpha")

    args = parser.parse_args()

    # 1. 加载模型
    MODEL = HF_NAMES.get(args.model_name, None)
    if MODEL is None:
        raise ValueError(f"不支持的模型名: {args.model_name}")

    print(f"Loading model: {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    dtype = torch.bfloat16 if 'llama3' in args.model_name else torch.float16
    model = llama.LlamaForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, torch_dtype=dtype, device_map='auto')
    device = model.device
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # 2. 准备数据
    print("Loading dataset...")
    inputs, gold_answers_list = build_nq_generation_inputs(
        args.dataset_path, tokenizer, max_samples=None, max_docs=args.max_docs,
        use_chat_template=args.use_chat_template, sample_size=args.sample_size, sample_seed=args.sample_seed,
    )
    print(f"Dataset size: {len(inputs)}")

    # 3. 运行 Baseline
    print("Running Baseline...")
    _, baseline_metrics = run_baseline(model, tokenizer, device, inputs, gold_answers_list, args.max_new_tokens)
    print(f"Baseline: EM={baseline_metrics['EM']:.4f}, F1={baseline_metrics['F1']:.4f}")

    # 4. 加载探针和特征
    print("Loading probes and features...")
    with open(args.probes_path, 'rb') as f:
        probes = pickle.load(f)
    val_accs = np.load(args.val_accs_path)
    tuning_headwise = np.load(args.tuning_headwise_path)
    tuning_labels = np.load(args.tuning_labels_path)
    
    # 计算 COM 方向
    B_th, L_th, HD_th = tuning_headwise.shape
    tuning_sep = rearrange(tuning_headwise, 'b l (h d) -> b l h d', h=args.num_heads, d=args.head_dim)
    num_questions = B_th // 2
    separated_head, separated_labels, _ = get_separated_activations_nq(tuning_labels, tuning_sep, num_questions)
    train_set_idxs = np.arange(num_questions)
    val_set_idxs = np.array([], dtype=int)
    com_directions = get_com_directions(L_th, args.num_heads, train_set_idxs, val_set_idxs, separated_head, separated_labels)

    # 5. 逐层干预实验
    results = []
    print(f"Starting Layer-wise Intervention (Alpha={args.alpha})...")
    
    # 我们遍历每一层
    for layer in tqdm(range(args.num_layers), desc="Layer Loop"):
        _, metrics = run_layer_intervention(
            model, tokenizer, device, inputs, gold_answers_list,
            layer_idx=layer,
            alpha=args.alpha,
            probes=probes,
            tuning_headwise=tuning_headwise,
            head_dim=args.head_dim,
            num_heads=args.num_heads,
            val_accs=val_accs,
            max_new_tokens=args.max_new_tokens,
            com_directions=com_directions
        )
        
        # 计算增长
        delta_em = metrics['EM'] - baseline_metrics['EM']
        delta_f1 = metrics['F1'] - baseline_metrics['F1']
        
        results.append({
            'layer': layer,
            'EM': metrics['EM'],
            'F1': metrics['F1'],
            'Delta_EM': delta_em,
            'Delta_F1': delta_f1
        })

    # 6. 排序并保存结果
    # 综合排序：这里简单地按 Delta_F1 + Delta_EM 排序，或者只按 F1
    # 题目要求“根据EM和F1分数相对于标准RAG的增长进行综合排序”
    # 我们定义 Score = Delta_EM + Delta_F1
    for r in results:
        r['Score'] = r['Delta_EM'] + r['Delta_F1']
        
    sorted_results = sorted(results, key=lambda x: x['Score'], reverse=True)

    print(f"Saving results to {args.output_csv}...")
    os.makedirs(os.path.dirname(args.output_csv) or '.', exist_ok=True)
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['layer', 'EM', 'F1', 'Delta_EM', 'Delta_F1', 'Score'])
        writer.writeheader()
        for r in sorted_results:
            writer.writerow(r)
            
    print("Done.")
    # 打印前5名
    print("Top 5 Layers:")
    for r in sorted_results[:5]:
        print(f"Layer {r['layer']}: ΔEM={r['Delta_EM']:.4f}, ΔF1={r['Delta_F1']:.4f}, Score={r['Score']:.4f}")

if __name__ == "__main__":
    main()
