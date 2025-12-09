import os
import torch
from tqdm import tqdm
import numpy as np
import pickle
import sys
sys.path.append('../')
from llama_utils import (
    get_llama_activations_bau,
    tokenized_nq_with_docs_dual,
)
import llama
import argparse
from transformers import AutoTokenizer

# 模型路径映射
HF_NAMES = {
    'llama2_chat_7B': '/root/shared-nvme/RAG-llm/models/Llama-2-7b-chat-hf', 
    'llama3_8B_instruct': '/root/shared-nvme/RAG-llm/models/Llama-3-8B-Instruct',
}

def main(): 
    """
    处理 NQ (RAG) 数据集，提取 Llama 模型的层级和头级激活特征，支持自定义输出目录
    """
    parser = argparse.ArgumentParser()
    # 模型相关参数
    parser.add_argument('--model_name', type=str, default='llama_7B', help='基础模型名称')
    parser.add_argument('--model_prefix', type=str, default='', help='模型名称前缀')
    # 数据集相关参数（仅支持 NQ）
    parser.add_argument('--dataset_name', type=str, default='nq', choices=['nq'], help='数据集名称（仅支持 nq）')
    default_nq_path = os.path.join(os.path.dirname(__file__), 'data/llama_2_chat_7b_train.jsonl')
    parser.add_argument('--nq_jsonl', type=str, default=default_nq_path, help='NQ 格式数据集路径')
    parser.add_argument('--nq_max_samples', type=int, default=None, help='仅处理指定数量的样本')
    parser.add_argument('--nq_max_docs', type=int, default=None, help='每条样本使用的检索片段数量上限')
    parser.add_argument('--use_chat_template', action='store_true', help='使用 tokenizer.apply_chat_template 构造聊天输入')
    # 输出配置
    parser.add_argument('--output_dir', type=str, default='../RAG-llm/RAG/features', help='输出特征文件的目录')
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

    # 处理 NQ 数据集，构造 prompts 和相关元数据
    print("Tokenizing NQ prompts...")
    prompts, labels, categories, tokens = tokenized_nq_with_docs_dual(
        args.nq_jsonl,
        tokenizer,
        max_samples=args.nq_max_samples,
        max_docs=args.nq_max_docs,
        use_chat_template=args.use_chat_template,
    )

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 保存元数据
    print("Saving NQ metadata...")
    with open(os.path.join(args.output_dir, f'{args.model_name}_{args.dataset_name}_categories.pkl'), 'wb') as f:
        pickle.dump(categories, f)
    with open(os.path.join(args.output_dir, f'{args.model_name}_{args.dataset_name}_tokens.pkl'), 'wb') as f:
        pickle.dump(tokens, f)

    # 提取激活特征
    print("Extracting model activations...")
    all_layer_wise_activations = []
    all_head_wise_activations = []
    for prompt in tqdm(prompts, desc="Processing prompts"):
        layer_activs, head_activs, _ = get_llama_activations_bau(model, prompt, device)
        # 仅保留最后一个 token 的激活特征
        all_layer_wise_activations.append(layer_activs[:, -1, :].copy())
        all_head_wise_activations.append(head_activs[:, -1, :].copy())

    # 保存激活特征和标签
    print("Saving output features...")
    np.save(os.path.join(args.output_dir, f'{args.model_name}_{args.dataset_name}_labels.npy'), labels)
    np.save(os.path.join(args.output_dir, f'{args.model_name}_{args.dataset_name}_layer_wise.npy'), all_layer_wise_activations)
    np.save(os.path.join(args.output_dir, f'{args.model_name}_{args.dataset_name}_head_wise.npy'), all_head_wise_activations)

    print(f"All features saved to: {args.output_dir}")

if __name__ == '__main__':
    main()