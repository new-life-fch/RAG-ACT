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
    _load_nq_jsonl,
    _build_messages_input,
    evaluate_nq_em_f1,
)
from utils.prompts_templates import prompt_dict


HF_NAMES = {
    'llama2_chat_7B': '/root/shared-nvme/RAG-llm/models/Llama-2-7b-chat-hf',
    'llama3_8B_instruct': '/root/shared-nvme/RAG-llm/models/Llama-3-8B-Instruct',
    'llama2_chat_13B': '/root/shared-nvme/RAG-llm/models/Llama-2-13b-chat-hf',
    'vicuna_7B_v1.5': '/root/shared-nvme/RAG-llm/models/vicuna-7b-v1.5',
}


def build_nq_con_notes_inputs(
    jsonl_path: str,
    tokenizer,
    max_samples: int = None,
    max_docs: int = None,
    use_chat_template: bool = True,
    sample_size: Optional[int] = None,
    sample_seed: int = 2025,
) -> Tuple[List[torch.Tensor], List[List[str]], List[Tuple[str, str]]]:
    entries = _load_nq_jsonl(jsonl_path, max_samples=max_samples)

    if sample_size is not None and sample_size > 0:
        rng = np.random.RandomState(sample_seed)
        idxs = rng.choice(len(entries), size=min(sample_size, len(entries)), replace=False)
        entries = [entries[i] for i in idxs]

    inputs: List[torch.Tensor] = []
    gold_answers_list: List[List[str]] = []
    metas: List[Tuple[str, str]] = []

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
        system_prompt = prompt_dict['qa']['CoN_notes_system']
        user_prompt = prompt_dict['qa']['CoN_notes_user'].format(passages=docs_block, question=question)

        input_ids = _build_messages_input(tokenizer, system_prompt, user_prompt, assistant_content=None, use_chat_template=use_chat_template)
        inputs.append(input_ids)
        gold_answers_list.append(list(answers))
        metas.append((docs_block, question))

    return inputs, gold_answers_list, metas


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


def run_generate_texts(model, tokenizer, device, inputs: List[torch.Tensor], max_new_tokens: int, desc: str) -> List[str]:
    outputs: List[str] = []
    with torch.no_grad():
        for input_ids in tqdm(inputs, desc=desc):
            input_ids = input_ids.to(device)
            gen_tokens = model.generate(
                input_ids,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
            )[:, input_ids.shape[-1]:]
            gen_str = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
            outputs.append(gen_str.strip())
    return outputs


def main():
    parser = argparse.ArgumentParser(description='NQ 标准RAG + CoN 提示词')
    parser.add_argument('--model_name', type=str, default='llama2_chat_7B')
    parser.add_argument('--dataset_path', type=str, default='/root/shared-nvme/RAG-llm/RAG/data/test.jsonl')
    parser.add_argument('--use_chat_template', action='store_true')
    parser.add_argument('--max_docs', type=int, default=None)
    parser.add_argument('--sample_size', type=int, default=100)
    parser.add_argument('--sample_seed', type=int, default=2025)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--results_root', type=str, default='./RAG/results_dump/llama-2-7b-instruct-con')

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

    notes_inputs, gold_answers_list, metas = build_nq_con_notes_inputs(
        args.dataset_path, tokenizer, max_samples=None, max_docs=args.max_docs,
        use_chat_template=args.use_chat_template, sample_size=args.sample_size, sample_seed=args.sample_seed,
    )
    notes_texts = run_generate_texts(model, tokenizer, device, notes_inputs, args.max_new_tokens, desc='con_notes_generate')

    answer_inputs: List[torch.Tensor] = []
    for (docs_block, question), notes in zip(metas, notes_texts):
        system_prompt = prompt_dict['qa']['CoN_answer_system']
        user_prompt = prompt_dict['qa']['CoN_answer_user'].format(passages=docs_block, notes=notes, question=question)
        input_ids = _build_messages_input(tokenizer, system_prompt, user_prompt, assistant_content=None, use_chat_template=args.use_chat_template)
        answer_inputs.append(input_ids)

    raw_answers = run_generate_texts(model, tokenizer, device, answer_inputs, args.max_new_tokens, desc='con_answer_generate')
    baseline_ans = [clean_answer_text(s) for s in raw_answers]
    em_b, f1_b = evaluate_nq_em_f1(baseline_ans, gold_answers_list)
    baseline_sum = {"EM": em_b, "F1": f1_b, "alpha": 0.0, "model_name": args.model_name, "intervention": "none", "sample_size": args.sample_size, "prompt": "CoN_two_stage"}

    baseline_ans_path = os.path.join(ans_dir, 'nq_con_two_stage_answers.jsonl')
    baseline_sum_path = os.path.join(sum_dir, 'nq_con_two_stage_summary.json')
    with open(baseline_ans_path, 'w', encoding='utf-8') as f:
        for pred, golds in zip(baseline_ans, gold_answers_list):
            f.write(json.dumps({"prediction": pred, "gold_answers": golds}, ensure_ascii=False) + '\n')
    with open(baseline_sum_path, 'w', encoding='utf-8') as f:
        json.dump(baseline_sum, f, ensure_ascii=False, indent=2)

    print(
        f"[CoN Two-Stage] samples={args.sample_size} EM={baseline_sum['EM']:.4f} "
        f"F1={baseline_sum['F1']:.4f}"
    )


if __name__ == '__main__':
    main()
