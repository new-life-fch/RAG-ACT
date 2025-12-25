import os
import argparse
import json
import pickle
import time
import torch
import numpy as np
from tqdm import tqdm
from einops import rearrange
from transformers import AutoTokenizer
from typing import List, Tuple, Dict, Optional

import llama
from baukit import TraceDict
from llama_utils import (
    _load_data_jsonl,
    _build_messages_input,
    get_interventions_dict,
    get_separated_activations_dataset,
    get_com_directions,
)
from utils.prompts_templates import prompt_dict
import csv

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
):
    entries = _load_data_jsonl(jsonl_path, max_samples=max_samples)

    if sample_size is not None and sample_size > 0:
        rng = np.random.RandomState(sample_seed)
        idxs = rng.choice(len(entries), size=min(sample_size, len(entries)), replace=False)
        entries = [entries[i] for i in idxs]

    inputs = []
    for ex in entries:
        question = ex["query"]
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

    return inputs

def load_scores_csv(path: str):
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        # 尝试检测并跳过表头
        try:
            first_row = next(reader)
            if not first_row[0].isdigit():
                pass # 是表头，跳过
            else:
                # 不是表头，处理第一行
                layer = int(first_row[0]); head = int(first_row[1]); score = float(first_row[2])
                items.append((layer, head, score))
        except StopIteration:
            return items
        except Exception:
            pass

        for row in reader:
            if not row: continue
            try:
                layer = int(row[0]); head = int(row[1]); score = float(row[2])
                items.append((layer, head, score))
            except Exception: continue
    return items

def main():
    parser = argparse.ArgumentParser(description='RAG 推理时间对比实验')
    parser.add_argument('--model_name', type=str, default='llama2_chat_7B')
    parser.add_argument('--dataset_path', type=str, default='RAG/data/PopQA/test_noise_test_noise4.jsonl')
    parser.add_argument('--use_chat_template', action='store_true')
    parser.add_argument('--max_docs', type=int, default=5)
    parser.add_argument('--sample_size', type=int, default=300)
    parser.add_argument('--head_dim', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=None)
    parser.add_argument('--probes_path', type=str, required=True)
    parser.add_argument('--val_accs_path', type=str, required=True)
    parser.add_argument('--tuning_headwise_path', type=str, required=True)
    parser.add_argument('--tuning_labels_path', type=str, required=True)
    parser.add_argument('--scores_csv', type=str, required=True)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--alpha', type=float, default=5.0)
    parser.add_argument('--num_intervention_heads', type=int, default=35)
    
    args = parser.parse_args()

    # 加载模型
    model_path = HF_NAMES.get(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dtype = torch.bfloat16 if 'llama3' in args.model_name else torch.float16
    model = llama.LlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, torch_dtype=dtype, device_map='auto')
    device = model.device
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 构造输入
    inputs = build_nq_generation_inputs(
        args.dataset_path,
        tokenizer,
        max_docs=args.max_docs,
        use_chat_template=True, # 明确开启，避免返回 None
        sample_size=args.sample_size
    )

    # 准备干预
    with open(args.probes_path, 'rb') as f:
        probes = pickle.load(f)
    val_accs = np.load(args.val_accs_path)
    tuning_headwise = np.load(args.tuning_headwise_path)
    tuning_labels = np.load(args.tuning_labels_path)
    scores = load_scores_csv(args.scores_csv)
    
    # 获取干预头
    by_score = sorted(scores, key=lambda x: x[2], reverse=True)
    top_heads = [(l, h) for (l, h, s) in by_score[:args.num_intervention_heads]]

    # 计算 com_directions
    B_th, L_th, HD_th = tuning_headwise.shape
    num_heads_calc = args.num_heads if args.num_heads else HD_th // args.head_dim
    tuning_sep = rearrange(tuning_headwise, 'b l (h d) -> b l h d', h=num_heads_calc, d=args.head_dim)
    num_questions = B_th // 2
    separated_head, separated_labels, _ = get_separated_activations_dataset(tuning_labels, tuning_sep, num_questions)
    com_directions = get_com_directions(L_th, num_heads_calc, np.arange(num_questions), np.array([]), separated_head, separated_labels)

    interventions = get_interventions_dict(
        top_heads, probes, tuning_sep, num_heads_calc,
        use_center_of_mass=True, use_random_dir=False,
        com_directions=com_directions, probe_score_map=val_accs
    )

    def lt_modulated_vector_add(head_output, layer_name):
        h_out = rearrange(head_output, 'b s (h d) -> b s h d', h=num_heads_calc)
        try:
            layer_idx = int(str(layer_name).split('.')[2])
        except:
            layer_idx = None
        for head, direction, proj_val_std, probe_factor in interventions[layer_name]:
            direction_to_add = torch.tensor(direction, dtype=h_out.dtype, device=h_out.device)
            strength = args.alpha * proj_val_std * float(probe_factor)
            h_out[:, -1, head, :] += strength * direction_to_add
        return rearrange(h_out, 'b s h d -> b s (h d)')

    layers_to_intervene = list(interventions.keys())

    # 预热 (Warm-up)
    print("\n开始 GPU 预热...")
    with torch.no_grad():
        for i in range(min(2, len(inputs))):
            input_ids = inputs[i].to(device)
            model.generate(input_ids, max_new_tokens=5)
    torch.cuda.synchronize()

    # 1. Naive RAG 推理时间
    print("\n开始测试 Naive RAG 推理时间...")
    torch.cuda.synchronize()
    start_naive = time.time()
    with torch.no_grad():
        for input_ids in tqdm(inputs, desc='Naive RAG'):
            input_ids = input_ids.to(device)
            model.generate(
                input_ids,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
    torch.cuda.synchronize()
    end_naive = time.time()
    naive_time = end_naive - start_naive

    # 2. Intervention RAG 推理时间
    print("\n开始测试 Intervention RAG 推理时间...")
    torch.cuda.synchronize()
    start_intervene = time.time()
    with torch.no_grad():
        for input_ids in tqdm(inputs, desc='Intervention RAG'):
            input_ids = input_ids.to(device)
            with TraceDict(model, layers_to_intervene, edit_output=lt_modulated_vector_add):
                model.generate(
                    input_ids,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                )
    torch.cuda.synchronize()
    end_intervene = time.time()
    intervene_time = end_intervene - start_intervene

    # 结果打印
    print("\n" + "="*50)
    print(f"实验结果 (样本数: {args.sample_size}, max_new_tokens: {args.max_new_tokens})")
    print(f"Naive RAG 总时间: {naive_time:.2f}s, 平均每样本: {naive_time/args.sample_size:.4f}s")
    print(f"Intervention RAG 总时间: {intervene_time:.2f}s, 平均每样本: {intervene_time/args.sample_size:.4f}s")
    print(f"推理开销 (Overhead): {(intervene_time/naive_time - 1)*100:.2f}%")
    print("="*50)

    # 保存结果
    results = {
        "sample_size": args.sample_size,
        "max_new_tokens": args.max_new_tokens,
        "naive_total_time": naive_time,
        "naive_avg_time": naive_time / args.sample_size,
        "intervention_total_time": intervene_time,
        "intervention_avg_time": intervene_time / args.sample_size,
        "overhead_percent": (intervene_time / naive_time - 1) * 100
    }
    with open("inference_time_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"结果已保存至 inference_time_results.json")

if __name__ == '__main__':
    from typing import Optional
    main()
