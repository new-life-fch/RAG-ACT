import os
import argparse
import json
import pickle
import torch
import numpy as np
from tqdm import tqdm
from einops import rearrange
from transformers import AutoTokenizer

import llama
from baukit import TraceDict
from llama_utils import (
    _load_data_jsonl,
    get_interventions_dict,
    evaluate_rag_em_f1,
    get_separated_activations_dataset,
    get_com_directions,
)
from generate import (
    HF_NAMES, 
    build_nq_generation_inputs, 
    load_scores_csv, 
    infer_LH_from_scores, 
    clean_answer_text,
    run_intervention,
    run_baseline
)

def get_ablation_heads(scores, L, H, mode, k=35, seed=2025):
    """根据消融模式选择注意力头"""
    by_score = sorted(scores, key=lambda x: x[2], reverse=True)
    all_heads = [(l, h) for l in range(L) for h in range(H)]
    
    if mode == 'bottom_heads':
        # 选择准确率最低的 k 个头
        return [(l, h) for (l, h, s) in by_score[-k:]]
    elif mode == 'random_heads':
        # 随机选择 k 个头
        rng = np.random.RandomState(seed)
        idxs = rng.choice(len(all_heads), size=k, replace=False)
        return [all_heads[i] for i in idxs]
    else:
        # 默认选择 Top-k 个头
        return [(l, h) for (l, h, s) in by_score[:k]]

def main():
    parser = argparse.ArgumentParser(description='RAG 消融实验脚本')
    parser.add_argument('--model_name', type=str, default='llama2_chat_7B')
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--ablation_mode', type=str, required=True, 
                        choices=['probe_coef', 'reverse_com', 'random_dir', 'random_heads', 'bottom_heads'],
                        help='消融实验模式')
    parser.add_argument('--alpha', type=float, default=7.0)
    parser.add_argument('--top_k', type=int, default=35)
    parser.add_argument('--use_chat_template', action='store_true')
    parser.add_argument('--sample_size', type=int, default=300)
    parser.add_argument('--probes_path', type=str, required=True)
    parser.add_argument('--val_accs_path', type=str, required=True)
    parser.add_argument('--tuning_headwise_path', type=str, required=True)
    parser.add_argument('--tuning_labels_path', type=str, required=True)
    parser.add_argument('--scores_csv', type=str, required=True)
    parser.add_argument('--results_root', type=str, default='./RAG/results/ablation')
    parser.add_argument('--max_new_tokens', type=int, default=256)
    
    args = parser.parse_args()
    
    # 1. 环境准备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    MODEL_PATH = HF_NAMES.get(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    dtype = torch.bfloat16 if 'llama3' in args.model_name else torch.float16
    model = llama.LlamaForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=dtype, device_map='auto')
    
    # 2. 加载数据
    inputs, gold_answers = build_nq_generation_inputs(
        args.dataset_path, tokenizer, sample_size=args.sample_size, use_chat_template=args.use_chat_template
    )
    
    # 3. 加载探针和激活
    with open(args.probes_path, 'rb') as f:
        probes = pickle.load(f)
    val_accs = np.load(args.val_accs_path)
    tuning_headwise = np.load(args.tuning_headwise_path)
    tuning_labels = np.load(args.tuning_labels_path)
    scores = load_scores_csv(args.scores_csv)
    L, H = infer_LH_from_scores(scores)
    
    # 4. 计算方向
    B, _, HD = tuning_headwise.shape
    head_dim = HD // H
    tuning_sep = rearrange(tuning_headwise, 'b l (h d) -> b l h d', h=H, d=head_dim)
    num_qs = B // 2
    sep_head, sep_labels, _ = get_separated_activations_dataset(tuning_labels, tuning_sep, num_qs)
    com_directions = get_com_directions(L, H, np.arange(num_qs), np.array([]), sep_head, sep_labels)
    
    # 5. 根据模式调整方向或选择头
    selected_heads = get_ablation_heads(scores, L, H, args.ablation_mode, k=args.top_k)
    
    current_com = com_directions.copy()
    use_com = True
    use_rand = False
    
    if args.ablation_mode == 'reverse_com':
        current_com = -current_com
    elif args.ablation_mode == 'random_dir':
        use_com = False
        use_rand = True
    elif args.ablation_mode == 'probe_coef':
        use_com = False
        use_rand = False
        
    # 6. 执行干预实验
    print(f"开始消融实验模式: {args.ablation_mode}, Alpha: {args.alpha}, Heads: {len(selected_heads)}")
    
    preds, metrics = run_intervention(
        model, tokenizer, device, inputs, gold_answers,
        alpha=args.alpha,
        strategy_name=args.ablation_mode,
        top_heads=selected_heads,
        probes=probes,
        tuning_headwise=tuning_headwise,
        head_dim=head_dim,
        num_heads=H,
        use_probe_factor=False,
        val_accs=val_accs,
        max_new_tokens=args.max_new_tokens,
        com_directions=current_com if use_com else None,
        use_center_of_mass=use_com,
        use_random_dir=use_rand
    )
    
    # 7. 保存结果
    res_dir = os.path.join(args.results_root, args.ablation_mode)
    os.makedirs(res_dir, exist_ok=True)
    
    output_file = os.path.join(res_dir, f"alpha_{args.alpha}_topk_{args.top_k}.json")
    with open(output_file, 'w') as f:
        json.dump({"metrics": metrics, "config": vars(args)}, f, indent=2)
    
    print(f"实验完成! EM: {metrics['EM']:.4f}, F1: {metrics['F1']:.4f}")
    print(f"结果已保存至: {output_file}")

if __name__ == "__main__":
    main()