import os
import argparse
import json
import pickle
from typing import List, Tuple

import torch
import numpy as np
from tqdm import tqdm
from einops import rearrange

import llama
from baukit import TraceDict
from transformers import AutoTokenizer
from llama_utils import (
    _load_nq_jsonl,
    _build_messages_input,
    get_interventions_dict,
    evaluate_nq_em_f1,
)


HF_NAMES = {
    'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf',
    'llama3_8B': '/root/shared-nvme/RAG-llm/models/Llama-3.1-8B-Instruct',
}


def build_nq_generation_inputs(
    jsonl_path: str,
    tokenizer,
    max_samples: int = None,
    max_docs: int = None,
    use_chat_template: bool = True,
    sample_size: int | None = None,
    sample_seed: int = 2025,
) -> Tuple[List[torch.Tensor], List[List[str]]]:
    """
    为 NQ 数据生成阶段构造输入：系统 + 用户消息（不包含助手答案）。
    返回：
    - inputs: 列表，元素为 `torch.Tensor` 的 `input_ids`
    - gold_answers_list: 列表，元素为正确答案字符串列表（用于 EM/F1）
    """
    entries = _load_nq_jsonl(jsonl_path, max_samples=max_samples)

    # 随机抽样（如果指定 sample_size）
    if sample_size is not None and sample_size > 0:
        rng = np.random.RandomState(sample_seed)
        idxs = rng.choice(len(entries), size=min(sample_size, len(entries)), replace=False)
        entries = [entries[i] for i in idxs]

    inputs: List[torch.Tensor] = []
    gold_answers_list: List[List[str]] = []

    for i, ex in enumerate(entries):
        # 字段检查
        if not all(k in ex for k in ["query", "answers", "retrieve_snippets"]):
            raise ValueError(f"new_dataset.jsonl 第 {i} 条记录缺少 query/answers/retrieve_snippets 字段")

        question = ex["query"]
        answers = ex["answers"]
        snippets = ex["retrieve_snippets"]

        # 取最多 max_docs 个片段文本
        docs_texts = []
        for j, snip in enumerate(snippets):
            if max_docs is not None and j >= max_docs:
                break
            text = snip.get("text", "")
            if isinstance(text, str) and len(text.strip()):
                docs_texts.append(text.strip())

        # 系统/用户提示词（强调只输出直接答案）
        docs_block = "\n\n".join([f"Document {k+1}: {d}" for k, d in enumerate(docs_texts)])
        system_prompt = (
            "Answer the question based on the given document. "
            "Provide only the most direct and concise answer. Do not include explanations, full sentences, or additional context. "
            "Just give the key information that directly answers the question.\n\n"
            "Example:\n"
            "Question: Where do the Great Lakes meet the ocean?\n"
            "Answer: the Saint Lawrence River\n\n"
            f"The following are given documents.\n\n{docs_block}"
        )
        user_prompt = f"Question: {question}\nAnswer:"

        # 注意：生成阶段不提供助手答案
        input_ids = _build_messages_input(tokenizer, system_prompt, user_prompt, assistant_content=None, use_chat_template=use_chat_template)
        inputs.append(input_ids)
        gold_answers_list.append(list(answers))

    return inputs, gold_answers_list


def main():
    """
    NQ 生成阶段脚本（随机方向 & 固定强度）：
    - 随机方向：在前48个头上添加随机方向（单位向量），强度按 proj_val_std 调制。
    - 固定强度：使用探针方向但不使用探针分数因子（probe_factor=1.0）。

    两者均采用：
    - 提示词使用chat模板（参照 utils/generate_dataset.py#L45-91）；
    - 贪心解码（do_sample=False，max_new_tokens控制长度）。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama2_chat_7B')
    parser.add_argument('--dataset_path', type=str, default='/root/shared-nvme/RAG-llm/RAG/data/test.jsonl')
    parser.add_argument('--use_chat_template', action='store_true')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--max_docs', type=int, default=None)
    parser.add_argument('--sample_size', type=int, default=300, help='随机抽取的样本数')
    parser.add_argument('--sample_seed', type=int, default=2025, help='随机抽样种子')
    parser.add_argument('--alpha', type=float, default=15.0, help='干预强度系数')
    parser.add_argument('--head_dim', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=None, help='若不传，将从特征维度推断')
    # 已保存的top-heads与探针（固定强度使用），随机方向也需要top-heads集合
    parser.add_argument('--top_heads_path', type=str, required=True)
    parser.add_argument('--probes_path', type=str, required=False, default=None)
    # 用于计算 proj_val_std 的激活（推荐使用 NQ 收集的激活）
    parser.add_argument('--tuning_headwise_path', type=str, default='../features/llama2_chat_7B_nq_head_wise.npy')
    parser.add_argument('--save_answers_random_path', type=str, default='./results_dump/answer_dump/nq_gen_answers_random_dir.jsonl')
    parser.add_argument('--save_summary_random_path', type=str, default='./results_dump/summary_dump/nq_gen_summary_random_dir.json')
    parser.add_argument('--save_answers_fixed_path', type=str, default='./results_dump/answer_dump/nq_gen_answers_fixed_strength.jsonl')
    parser.add_argument('--save_summary_fixed_path', type=str, default='./results_dump/summary_dump/nq_gen_summary_fixed_strength.json')
    parser.add_argument('--max_new_tokens', type=int, default=256, help='生成最大新token数（贪心解码）')
    args = parser.parse_args()

    MODEL = HF_NAMES.get(args.model_name, None)
    if MODEL is None:
        raise ValueError(f"不支持的模型名: {args.model_name}")

    # 加载分词器与模型
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    dtype = torch.bfloat16 if 'llama3' in args.model_name else torch.float16
    model = llama.LlamaForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, torch_dtype=dtype, device_map='auto')
    device = model.device
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # 构造生成输入与金标准答案列表
    inputs, gold_answers_list = build_nq_generation_inputs(
        args.dataset_path,
        tokenizer,
        max_samples=args.max_samples,
        max_docs=args.max_docs,
        use_chat_template=args.use_chat_template,
        sample_size=args.sample_size,
        sample_seed=args.sample_seed,
    )

    # 加载 top-heads
    with open(args.top_heads_path, 'rb') as f:
        top_heads = pickle.load(f)  # List[Tuple[layer, head]]

    # 加载用于计算 proj_val_std 的 head-wise 激活，并 reshape 为 (B, L, H, D)
    tuning_headwise = np.load(args.tuning_headwise_path)  # (B, L, H*D)
    B, L, HD = tuning_headwise.shape
    if args.num_heads is None:
        if HD % args.head_dim != 0:
            raise ValueError('无法从特征维推断 num_heads，请显式传入 --num_heads')
        num_heads = HD // args.head_dim
    else:
        num_heads = args.num_heads
    tuning_headwise = rearrange(tuning_headwise, 'b l (h d) -> b l h d', h=num_heads, d=args.head_dim)

    # 若固定强度使用探针方向，需要加载 probes
    probes = None
    if args.probes_path is not None:
        with open(args.probes_path, 'rb') as f:
            probes = pickle.load(f)

    # 构造干预函数
    def make_lt_add(interventions):
        def lt_modulated_vector_add(head_output, layer_name, start_edit_location='lt'):
            ho = rearrange(head_output, 'b s (h d) -> b s h d', h=num_heads)
            for head, direction, proj_val_std, probe_factor in interventions[layer_name]:
                direction_to_add = torch.tensor(direction).to(ho.device)
                if start_edit_location == 'lt':
                    ho[:, -1, head, :] += args.alpha * proj_val_std * probe_factor * direction_to_add
                else:
                    ho[:, start_edit_location:, head, :] += args.alpha * proj_val_std * probe_factor * direction_to_add
            ho = rearrange(ho, 'b s h d -> b s (h d)')
            return ho
        return lt_modulated_vector_add

    # 评估：随机方向 & 固定强度
    os.makedirs(os.path.dirname(args.save_answers_random_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.save_summary_random_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.save_answers_fixed_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.save_summary_fixed_path), exist_ok=True)

    preds_random = []
    preds_fixed = []

    def _clean_answer_text(gen_str: str) -> str:
        s = gen_str.strip()
        if 'Q:' in s:
            s = s.split('Q:')[0].strip()
        if 'Answer:' in s:
            try:
                s = s.split('Answer:')[1].strip()
            except Exception:
                pass
        return s

    # 干预字典：随机方向（probe_factor固定为1.0）
    interventions_random = get_interventions_dict(
        top_heads,
        probes,  # 未使用
        tuning_headwise,
        num_heads,
        use_center_of_mass=False,
        use_random_dir=True,
        com_directions=None,
        probe_score_map=None,
    )
    lt_add_random = make_lt_add(interventions_random)

    # 干预字典：固定强度（探针方向，无探针分数因子）
    if probes is None:
        raise ValueError('固定强度需要提供 --probes_path 以加载探针方向')
    interventions_fixed = get_interventions_dict(
        top_heads,
        probes,
        tuning_headwise,
        num_heads,
        use_center_of_mass=False,
        use_random_dir=False,
        com_directions=None,
        probe_score_map=None,  # 禁用探针分数因子
    )
    lt_add_fixed = make_lt_add(interventions_fixed)

    with torch.no_grad():
        for input_ids, golds in tqdm(zip(inputs, gold_answers_list), total=len(inputs), desc='nq_generate_random_fixed'):
            input_ids = input_ids.to(device)

            # 随机方向（贪心解码）
            layers_to_intervene = list(interventions_random.keys())
            with TraceDict(model, layers_to_intervene, edit_output=lt_add_random):
                gen_tokens_rand = model.generate(
                    input_ids,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id,
                )[:, input_ids.shape[-1]:]
            gen_str_rand = tokenizer.decode(gen_tokens_rand[0], skip_special_tokens=True)
            preds_random.append(_clean_answer_text(gen_str_rand))

            # 固定强度（贪心解码）
            layers_to_intervene = list(interventions_fixed.keys())
            with TraceDict(model, layers_to_intervene, edit_output=lt_add_fixed):
                gen_tokens_fix = model.generate(
                    input_ids,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id,
                )[:, input_ids.shape[-1]:]
            gen_str_fix = tokenizer.decode(gen_tokens_fix[0], skip_special_tokens=True)
            preds_fixed.append(_clean_answer_text(gen_str_fix))

    # 保存逐条预测
    with open(args.save_answers_random_path, 'w', encoding='utf-8') as f:
        for pred, golds in zip(preds_random, gold_answers_list):
            item = {"prediction": pred, "gold_answers": golds}
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    with open(args.save_answers_fixed_path, 'w', encoding='utf-8') as f:
        for pred, golds in zip(preds_fixed, gold_answers_list):
            item = {"prediction": pred, "gold_answers": golds}
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # 计算并保存 EM/F1
    em_r, f1_r = evaluate_nq_em_f1(preds_random, gold_answers_list)
    summary_r = {"EM": em_r, "F1": f1_r, "alpha": args.alpha, "model_name": args.model_name, "intervention": "random_dir_top48"}
    with open(args.save_summary_random_path, 'w', encoding='utf-8') as f:
        json.dump(summary_r, f, ensure_ascii=False, indent=2)

    em_f, f1_f = evaluate_nq_em_f1(preds_fixed, gold_answers_list)
    summary_f = {"EM": em_f, "F1": f1_f, "alpha": args.alpha, "model_name": args.model_name, "intervention": "probe_top48_fixed_strength"}
    with open(args.save_summary_fixed_path, 'w', encoding='utf-8') as f:
        json.dump(summary_f, f, ensure_ascii=False, indent=2)

    print("Random Direction Summary:", summary_r)
    print("Fixed Strength Summary:", summary_f)


if __name__ == '__main__':
    main()