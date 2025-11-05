import os
import sys
import argparse
import json
import pickle
from typing import List, Tuple, Any

import torch
import numpy as np
from tqdm import tqdm
from einops import rearrange

import llama
from baukit import TraceDict
from llama_utils import (
    _load_nq_jsonl,
    _build_messages_input,
    get_interventions_dict,
    evaluate_nq_em_f1,
)


HF_NAMES = {
    'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf',
    'llama3_8B': '/root/autodl-tmp/RAG-llm/models/Llama-3.1-8B-Instruct',
}


def build_nq_generation_inputs(
    jsonl_path: str,
    tokenizer,
    max_samples: int = None,
    max_docs: int = None,
    use_chat_template: bool = True,
) -> Tuple[List[torch.Tensor], List[List[str]]]:
    """
    为 NQ 数据生成阶段构造输入：系统 + 用户消息（不包含助手答案）。
    返回：
    - inputs: 列表，元素为 `torch.Tensor` 的 `input_ids`
    - gold_answers_list: 列表，元素为正确答案字符串列表（用于 EM/F1）

    初学者注释：生成阶段我们只给模型问题和检索到的文档信息，不提供答案，让模型自己生成；
    之后我们将生成的答案与所有正确答案列表进行比对，计算 EM/F1。
    """
    entries = _load_nq_jsonl(jsonl_path, max_samples=max_samples)

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

        if i == 0:
            print(f"[Generation Chat Input Example]\nSYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}")

    return inputs, gold_answers_list


def main():
    """
    NQ 生成阶段脚本：
    1) 加载模型与分词器；
    2) 从 jsonl 构造系统+用户输入（不含答案）；
    3) 加载并构造干预（基于已保存的探针与 top-k 头、验证准确率作为探针分数因子）；
    4) 在生成时应用干预；
    5) 评估 EM/F1（考虑全部正确答案列表），并保存结果。

    初学者注释：干预在注意力头的输出上加一个“方向向量”，让模型更倾向于生成正确答案；强度由三个因素共同决定：
    - `alpha`（你的手动设置）、
    - 沿该方向的激活标准差 `proj_val_std`（数据驱动的尺度）、
    - 探针分数因子 `probe_factor`（验证准确率，越高说明该头在区分正/负上更可靠）。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama2_chat_7B')
    parser.add_argument('--dataset_path', type=str, default='./new_dataset.jsonl')
    parser.add_argument('--use_chat_template', action='store_true')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--max_docs', type=int, default=None)
    parser.add_argument('--alpha', type=float, default=15.0, help='干预强度系数')
    parser.add_argument('--head_dim', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=None, help='若不传，将从特征维度推断')
    # 已保存的探针与top-k头、验证准确率
    parser.add_argument('--probes_path', type=str, required=True)
    parser.add_argument('--top_heads_path', type=str, required=True)
    parser.add_argument('--val_accs_path', type=str, required=True)
    # 用于计算 proj_val_std 的激活（推荐使用 NQ 收集的激活）
    parser.add_argument('--tuning_headwise_path', type=str, default='../features/llama2_chat_7B_nq_head_wise.npy')
    parser.add_argument('--tuning_labels_path', type=str, default='../features/llama2_chat_7B_nq_labels.npy')
    parser.add_argument('--save_answers_path', type=str, default='./results_dump/answer_dump/nq_gen_answers.jsonl')
    parser.add_argument('--save_summary_path', type=str, default='./results_dump/summary_dump/nq_gen_summary.json')
    args = parser.parse_args()

    MODEL = HF_NAMES.get(args.model_name, None)
    if MODEL is None:
        raise ValueError(f"不支持的模型名: {args.model_name}")

    # 加载分词器与模型
    tokenizer = llama.LlamaTokenizer.from_pretrained(MODEL)
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
    )

    # 加载 top-k 探针与验证准确率（作为探针分数因子）
    with open(args.probes_path, 'rb') as f:
        probes = pickle.load(f)
    with open(args.top_heads_path, 'rb') as f:
        top_heads = pickle.load(f)  # List[Tuple[layer, head]]
    val_accs = np.load(args.val_accs_path)  # shape (L, H)

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

    # 构造干预字典（包含探针分数因子）
    probe_score_map = val_accs  # (L, H)
    interventions = get_interventions_dict(
        top_heads,
        probes,
        tuning_headwise,
        num_heads,
        use_center_of_mass=False,
        use_random_dir=False,
        com_directions=None,
        probe_score_map=probe_score_map,
    )

    # 干预函数：在最后一个 token 的 head 输出上加方向
    def lt_modulated_vector_add(head_output, layer_name, start_edit_location='lt'):
        head_output = rearrange(head_output, 'b s (h d) -> b s h d', h=num_heads)
        for head, direction, proj_val_std, probe_factor in interventions[layer_name]:
            direction_to_add = torch.tensor(direction).to(head_output.device)
            if start_edit_location == 'lt':
                head_output[:, -1, head, :] += args.alpha * proj_val_std * probe_factor * direction_to_add
            else:
                head_output[:, start_edit_location:, head, :] += args.alpha * proj_val_std * probe_factor * direction_to_add
        head_output = rearrange(head_output, 'b s h d -> b s (h d)')
        return head_output

    # 逐条生成并评估
    os.makedirs(os.path.dirname(args.save_answers_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.save_summary_path), exist_ok=True)
    predictions = []

    with torch.no_grad():
        for input_ids, golds in tqdm(zip(inputs, gold_answers_list), total=len(inputs), desc='nq_generate'):
            max_len = input_ids.shape[-1] + 32  # 输出最多 32 个新 token，可按需调整
            layers_to_intervene = list(interventions.keys())
            with TraceDict(model, layers_to_intervene, edit_output=lt_modulated_vector_add) as ret:
                input_ids = input_ids.to(device)
                gen_tokens = model.generate(input_ids, max_length=max_len, num_return_sequences=1,)[:, input_ids.shape[-1]:]
            gen_str = tokenizer.decode(gen_tokens[0], skip_special_tokens=True).strip()
            # 简单清理：若模型误生成下一个问答模板，截断到下一个 "Q:" 之前
            if 'Q:' in gen_str:
                gen_str = gen_str.split('Q:')[0].strip()
            # 若再次出现 "Answer:" 模板，取其后内容
            if 'Answer:' in gen_str:
                try:
                    gen_str = gen_str.split('Answer:')[1].strip()
                except Exception:
                    pass

            predictions.append(gen_str)

    # 保存逐条预测
    with open(args.save_answers_path, 'w', encoding='utf-8') as f:
        for pred, golds in zip(predictions, gold_answers_list):
            item = {"prediction": pred, "gold_answers": golds}
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # 计算并保存 EM/F1
    em, f1 = evaluate_nq_em_f1(predictions, gold_answers_list)
    summary = {"EM": em, "F1": f1, "alpha": args.alpha, "model_name": args.model_name}
    with open(args.save_summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Summary:", summary)


if __name__ == '__main__':
    main()