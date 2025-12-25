import os
import argparse
import json
from typing import List, Tuple, Optional

import torch
import numpy as np
from tqdm import tqdm

import llama
from transformers import AutoTokenizer
from llama_utils import (
    _load_data_jsonl,
    _build_messages_input,
    evaluate_rag_em_f1,
)
from utils.prompts_templates import prompt_dict


HF_NAMES = {
    'llama2_chat_7B': '/root/shared-nvme/RAG-llm/models/Llama-2-7b-chat-hf',
    'llama3_8B_instruct': '/root/shared-nvme/RAG-llm/models/Llama-3-8B-Instruct',
    'llama2_chat_13B': '/root/shared-nvme/RAG-llm/models/Llama-2-13b-chat-hf',
    'vicuna_7B_v1.5': '/root/shared-nvme/RAG-llm/models/vicuna-7b-v1.5',
}


def build_nq_naive_inputs(
    jsonl_path: str,
    tokenizer,
    max_samples: int = None,
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
        if not all(k in ex for k in ["query", "answers"]):
            raise ValueError(f"JSONL 第 {i} 条记录缺少 query/answers 字段")

        question = ex["query"]
        answers = ex["answers"]

        system_prompt = prompt_dict['vicuna']['naive_LLM_system']
        user_prompt = prompt_dict['vicuna']['naive_LLM_user'].format(question=question)

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


def run_generate(model, tokenizer, device, inputs: List[torch.Tensor], max_new_tokens: int) -> List[str]:
    outputs: List[str] = []
    is_first_iter = True
    with torch.no_grad():
        for input_ids in tqdm(inputs, desc='naive_llm_generate'):
            if input_ids is None or input_ids.shape[-1] > 4096:
                raise ValueError(f"输入上下文 token 数超限: {0 if input_ids is None else input_ids.shape[-1]}，最大允许 4096")
            input_ids = input_ids.to(device)
            
            if is_first_iter:
                print("==== PROMPT ====")
                print(tokenizer.decode(input_ids[0]))
                # 标志置为 False，后续不再打印
                is_first_iter = False
            
            gen_tokens = model.generate(
                input_ids,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
            )[:, input_ids.shape[-1]:]
            gen_str = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
            outputs.append(clean_answer_text(gen_str))
    return outputs


def main():
    parser = argparse.ArgumentParser(description='NQ 原生大模型（naive LLM）不使用检索片段')
    parser.add_argument('--model_name', type=str, default='llama2_chat_7B')
    parser.add_argument('--dataset_path', type=str, default='/root/shared-nvme/RAG-llm/RAG/data/test.jsonl')
    parser.add_argument('--use_chat_template', action='store_true')
    parser.add_argument('--sample_size', type=int, default=100)
    parser.add_argument('--sample_seed', type=int, default=2025)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--results_root', type=str, default='./RAG/results_dump/llama-2-7b-naive-llm')

    args = parser.parse_args()

    ans_dir = os.path.join(args.results_root, 'answer_dump')
    sum_dir = os.path.join(args.results_root, 'summary_dump')
    os.makedirs(ans_dir, exist_ok=True)
    os.makedirs(sum_dir, exist_ok=True)

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

    inputs, gold_answers_list = build_nq_naive_inputs(
        args.dataset_path, tokenizer, max_samples=None,
        use_chat_template=args.use_chat_template, sample_size=args.sample_size, sample_seed=args.sample_seed,
    )

    preds = run_generate(model, tokenizer, device, inputs, args.max_new_tokens)
    em_b, f1_b = evaluate_rag_em_f1(preds, gold_answers_list)
    summary = {"EM": em_b, "F1": f1_b, "model_name": args.model_name, "sample_size": args.sample_size, "prompt": "naive_LLM"}

    ans_path = os.path.join(ans_dir, 'nq_naive_llm_answers.jsonl')
    sum_path = os.path.join(sum_dir, 'nq_naive_llm_summary.json')
    with open(ans_path, 'w', encoding='utf-8') as f:
        for pred, golds in zip(preds, gold_answers_list):
            f.write(json.dumps({"prediction": pred, "gold_answers": golds}, ensure_ascii=False) + '\n')
    with open(sum_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(
        f"[Naive LLM] samples={args.sample_size} EM={summary['EM']:.4f} "
        f"F1={summary['F1']:.4f}"
    )


if __name__ == '__main__':
    main()
