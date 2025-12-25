import os
import torch
from tqdm import tqdm
import numpy as np
import pickle
import sys
sys.path.append('../')
from llama_utils import (
    get_llama_activations_bau,
    tokenized_dataset_with_docs_dual,
    tokenized_dataset_noise_contrastive,
    _load_data_jsonl,
    _build_messages_input,
    prompt_dict
)
import llama
import argparse
from transformers import AutoTokenizer

# 模型路径映射
HF_NAMES = {
    'llama2_chat_7B': '/root/shared-nvme/RAG-llm/models/Llama-2-7b-chat-hf',
    'llama3_8B_instruct': '/root/shared-nvme/RAG-llm/models/Llama-3-8B-Instruct',
    'llama2_chat_13B': '/root/shared-nvme/RAG-llm/models/Llama-2-13b-chat-hf',
    'vicuna_7B_v1.5': '/root/shared-nvme/RAG-llm/models/vicuna-7b-v1.5',
}

def main(): 
    """
    处理 (RAG) 数据集，提取 Llama 模型的层级和头级激活特征，支持自定义输出目录
    """
    parser = argparse.ArgumentParser()
    # 模型相关参数
    parser.add_argument('--model_name', type=str, default='llama_7B', help='基础模型名称')
    parser.add_argument('--model_prefix', type=str, default='', help='模型名称前缀')
    # 数据集相关参数
    parser.add_argument('--dataset_name', type=str, default='nq', choices=['nq', 'triviaqa', 'popqa'], help='数据集名称')
    parser.add_argument('--data_jsonl', type=str, default=None, help='数据集路径')
    parser.add_argument('--data_max_samples', type=int, default=None, help='仅处理指定数量的样本')
    parser.add_argument('--data_max_docs', type=int, default=None, help='每条样本使用的检索片段数量上限')
    parser.add_argument('--use_chat_template', action='store_true', help='使用 tokenizer.apply_chat_template 构造聊天输入')
    parser.add_argument('--use_noise_contrastive', action='store_true', help='使用 Clean vs Noisy 对比数据，用于提取抗噪方向')
    # 输出配置
    parser.add_argument('--output_dir', type=str, default='../RAG-llm/RAG/features', help='输出特征文件的目录')
    # 提取位置配置
    parser.add_argument('--extraction_point', type=str, default='prompt_end', choices=['answer_end', 'prompt_end'], 
                        help='提取激活特征的位置：answer_end (整个序列最后) 或 prompt_end (Assistant 开始回答前)')
    # 设备配置
    parser.add_argument('--device', type=int, default=0, help='GPU 设备编号')
    args = parser.parse_args()

    # 加载模型和分词器
    model_name_or_path = HF_NAMES[args.model_prefix + args.model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = llama.LlamaForCausalLM.from_pretrained(
        model_name_or_path, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    device = model.device  # 与模型的 device_map 保持一致

    # 处理数据集，构造 prompts 和相关元数据
    print(f"Tokenizing NQ prompts (noise_contrastive={args.use_noise_contrastive})...")
    
    if args.use_noise_contrastive:
        prompts, labels, categories, tokens = tokenized_dataset_noise_contrastive(
            args.data_jsonl,
            tokenizer,
            max_samples=args.data_max_samples,
            max_docs=args.data_max_docs,
            use_chat_template=args.use_chat_template,
        )
    else:
        prompts, labels, categories, tokens = tokenized_dataset_with_docs_dual(
            args.data_jsonl,
            tokenizer,
            max_samples=args.data_max_samples,
            max_docs=args.data_max_docs,
            use_chat_template=args.use_chat_template,
        )

    # 如果是 prompt_end，需要知道 prompt 的结束位置
    prompt_lengths = []
    if args.extraction_point == 'prompt_end':
        print("Calculating prompt lengths for 'prompt_end' extraction...")
        entries = _load_data_jsonl(args.data_jsonl, max_samples=args.data_max_samples)
        
        for ex in entries:
            if not all(k in ex for k in ["query", "answers", "retrieve_snippets"]):
                continue
            
            question = ex["query"]
            snippets = ex["retrieve_snippets"]
            
            if args.use_noise_contrastive:
                # 1. Clean prompt length
                gold_snippet = snippets[0].get("text", "").strip()
                if not gold_snippet: continue
                clean_docs_block = f"Passage-1: {gold_snippet}"
                system_prompt = prompt_dict['qa']['RAG_system']
                clean_user_prompt = prompt_dict['qa']['RAG_user'].format(paras=clean_docs_block, question=question, answer='')
                p_ids_clean = _build_messages_input(tokenizer, system_prompt, clean_user_prompt, None, args.use_chat_template)
                prompt_lengths.append(p_ids_clean.shape[-1])
                
                # 2. Noisy prompt length
                docs_texts = [snip.get("text", "").strip() for snip in snippets[:args.data_max_docs] if snip.get("text", "").strip()]
                noisy_docs_block = "\n".join([f"Passage-{k+1}: {d}" for k, d in enumerate(docs_texts)])
                noisy_user_prompt = prompt_dict['qa']['RAG_user'].format(paras=noisy_docs_block, question=question, answer='')
                p_ids_noisy = _build_messages_input(tokenizer, system_prompt, noisy_user_prompt, None, args.use_chat_template)
                prompt_lengths.append(p_ids_noisy.shape[-1])
            else:
                docs_texts = [snip.get("text", "").strip() for snip in snippets[:args.data_max_docs] if snip.get("text", "").strip()]
                docs_block = "\n".join([f"Passage-{k+1}: {d}" for k, d in enumerate(docs_texts)])
                system_prompt = prompt_dict['qa']['RAG_system']
                user_prompt = prompt_dict['qa']['RAG_user'].format(paras=docs_block, question=question, answer='')
                p_ids = _build_messages_input(tokenizer, system_prompt, user_prompt, None, args.use_chat_template)
                p_len = p_ids.shape[-1]
                prompt_lengths.append(p_len)
                prompt_lengths.append(p_len)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 保存元数据
    print("Saving NQ metadata...")
    with open(os.path.join(args.output_dir, f'{args.model_name}_{args.dataset_name}_categories.pkl'), 'wb') as f:
        pickle.dump(categories, f)
    with open(os.path.join(args.output_dir, f'{args.model_name}_{args.dataset_name}_tokens.pkl'), 'wb') as f:
        pickle.dump(tokens, f)

    # 提取激活特征
    print(f"Extracting model activations at {args.extraction_point}...")
    all_layer_wise_activations = []
    all_head_wise_activations = []
    for i, prompt in enumerate(tqdm(prompts, desc="Processing prompts")):
        layer_activs, head_activs, _ = get_llama_activations_bau(model, prompt, device)
        
        if args.extraction_point == 'answer_end':
            idx = -1
        else:
            # prompt_end: 对应 prompt 的最后一个 token
            idx = prompt_lengths[i] - 1
            
        all_layer_wise_activations.append(layer_activs[:, idx, :].copy())
        all_head_wise_activations.append(head_activs[:, idx, :].copy())

    # 保存激活特征和标签
    print("Saving output features...")
    np.save(os.path.join(args.output_dir, f'{args.model_name}_{args.dataset_name}_labels.npy'), labels)
    np.save(os.path.join(args.output_dir, f'{args.model_name}_{args.dataset_name}_layer_wise.npy'), all_layer_wise_activations)
    np.save(os.path.join(args.output_dir, f'{args.model_name}_{args.dataset_name}_head_wise.npy'), all_head_wise_activations)

    print(f"All features saved to: {args.output_dir}")

if __name__ == '__main__':
    main()