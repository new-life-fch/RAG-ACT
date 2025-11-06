# Custom llama loading of getting activations (with head_out)
import os
import torch
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import pickle
import sys
sys.path.append('../')
from llama_utils import (
    get_llama_activations_bau,
    tokenized_tqa,
    tokenized_tqa_gen,
    tokenized_tqa_gen_end_q,
    tokenized_nq_with_docs_dual,
)
import llama
import pickle
import argparse
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import os

HF_NAMES = {
    # 'llama_7B': 'baffo32/decapoda-research-llama-7B-hf',
    'llama_7B': 'huggyllama/llama-7b',
    'alpaca_7B': 'circulus/alpaca-7b', 
    'vicuna_7B': 'AlekseyKorshuk/vicuna-7b', 
    'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf', 
    'llama2_chat_13B': 'meta-llama/Llama-2-13b-chat-hf', 
    'llama2_chat_70B': 'meta-llama/Llama-2-70b-chat-hf', 
    'llama3_8B': 'meta-llama/Meta-Llama-3-8B',
    'llama3_8B_instruct': '/root/shared-nvme/RAG-llm/models/Llama-3.1-8B-Instruct',
    'llama3_70B': 'meta-llama/Meta-Llama-3-70B',
    'llama3_70B_instruct': 'meta-llama/Meta-Llama-3-70B-Instruct'
}

def main(): 
    """
    Specify dataset name as the first command line argument. Current options are 
    "tqa_mc2", "piqa", "rte", "boolq", "copa". Gets activations for all prompts in the 
    validation set for the specified dataset on the last token for llama-7B. 
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama_7B')
    parser.add_argument('--model_prefix', type=str, default='', help='prefix of model name')
    parser.add_argument('--dataset_name', type=str, default='tqa_mc2')
# =====================
# NQ (RAG) 数据集支持
# =====================
    # 默认路径定位到当前脚本同目录下的 new_dataset.jsonl
    default_nq_path = os.path.join(os.path.dirname(__file__), 'new_dataset.jsonl')
    parser.add_argument('--nq_jsonl', type=str, default=default_nq_path, help='NQ 格式数据集路径（当 dataset_name=nq 时生效）')
    parser.add_argument('--nq_max_samples', type=int, default=None, help='NQ：仅处理指定数量的样本')
    parser.add_argument('--nq_max_docs', type=int, default=None, help='NQ：每条样本使用的检索片段数量上限')
    parser.add_argument('--use_chat_template', action='store_true', help='NQ：使用 tokenizer.apply_chat_template 构造系统+用户聊天输入')
# =====================
# NQ (RAG) 数据集支持
# =====================
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    model_name_or_path = HF_NAMES[args.model_prefix + args.model_name]

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # model = AutoModelForCausalLM.from_pretrained(model_name_or_path, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto")
    # tokenizer = llama.LlamaTokenizer.from_pretrained(model_name_or_path)
    model = llama.LlamaForCausalLM.from_pretrained(model_name_or_path, dtype=torch.float16, device_map="auto")
    # 更稳健的设备选择（与模型的 device_map 保持一致）
    device = model.device

    if args.dataset_name == "tqa_mc2": 
        dataset = load_dataset("truthfulqa/truthful_qa", "multiple_choice")['validation']
        formatter = tokenized_tqa
    elif args.dataset_name == "tqa_gen": 
        dataset = load_dataset("truthfulqa/truthful_qa", 'generation')['validation']
        formatter = tokenized_tqa_gen
    elif args.dataset_name == 'tqa_gen_end_q': 
        dataset = load_dataset("truthfulqa/truthful_qa", 'generation')['validation']
        formatter = tokenized_tqa_gen_end_q
# =====================
# NQ (RAG) 数据集支持
# =====================
    elif args.dataset_name == 'nq':
        dataset = None
        formatter = None
# =====================
# NQ (RAG) 数据集支持
# =====================
    else: 
        raise ValueError("Invalid dataset name")

    print("Tokenizing prompts")
    if args.dataset_name in ("tqa_gen", "tqa_gen_end_q"):
        prompts, labels, categories = formatter(dataset, tokenizer)
        os.makedirs('../features', exist_ok=True)
        with open(f'../features/{args.model_name}_{args.dataset_name}_categories.pkl', 'wb') as f:
            pickle.dump(categories, f)
    elif args.dataset_name == 'tqa_mc2':
        prompts, labels = formatter(dataset, tokenizer)
# =====================
# NQ (RAG) 数据集支持
# =====================
    elif args.dataset_name == 'nq':
        # NQ 流程：从 jsonl 读取，构造系统+用户+助手（teacher forcing）聊天输入
        prompts, labels, categories, tokens = tokenized_nq_with_docs_dual(
            args.nq_jsonl,
            tokenizer,
            max_samples=args.nq_max_samples,
            max_docs=args.nq_max_docs,
            use_chat_template=args.use_chat_template,
        )
        os.makedirs('../RAG-llm/features', exist_ok=True)
        with open(f'../RAG-llm/features/{args.model_name}_{args.dataset_name}_categories.pkl', 'wb') as f:
            pickle.dump(categories, f)
        with open(f'../RAG-llm/features/{args.model_name}_{args.dataset_name}_tokens.pkl', 'wb') as f:
            pickle.dump(tokens, f)
    else:
        raise ValueError("Invalid dataset name")
# =====================
# NQ (RAG) 数据集支持
# =====================

    all_layer_wise_activations = []
    all_head_wise_activations = []

    print("Getting activations")
    for prompt in tqdm(prompts):
        layer_wise_activations, head_wise_activations, _ = get_llama_activations_bau(model, prompt, device)
        all_layer_wise_activations.append(layer_wise_activations[:,-1,:].copy())
        all_head_wise_activations.append(head_wise_activations[:,-1,:].copy())

    print("Saving labels")
    np.save(f'../RAG-llm/features/{args.model_name}_{args.dataset_name}_labels.npy', labels)

    print("Saving layer wise activations")
    np.save(f'../RAG-llm/features/{args.model_name}_{args.dataset_name}_layer_wise.npy', all_layer_wise_activations)
    
    print("Saving head wise activations")
    np.save(f'../RAG-llm/features/{args.model_name}_{args.dataset_name}_head_wise.npy', all_head_wise_activations)

if __name__ == '__main__':
    main()
