import os
import sys
import argparse
import json
import pickle
from typing import List, Tuple, Any, Optional

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
    get_separated_activations_nq,
    get_com_directions,
)
import textwrap


HF_NAMES = {
    'llama2_chat_7B': '/root/shared-nvme/RAG-llm/models/Llama-2-7b-chat-hf',
    'llama3_8B_instruct': '/root/shared-nvme/RAG-llm/models/Llama-3.1-8B-Instruct',
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
    """
    为 NQ 数据生成阶段构造输入：系统 + 用户消息（不包含助手答案）。
    返回：
    - inputs: 列表，元素为 `torch.Tensor` 的 `input_ids`
    - gold_answers_list: 列表，元素为正确答案字符串列表（用于 EM/F1）

    初学者注释：生成阶段我们只给模型问题和检索到的文档信息，不提供答案，让模型自己生成；
    之后我们将生成的答案与所有正确答案列表进行比对，计算 EM/F1。
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
        system_prompt = textwrap.dedent(f'''
Answer the question based strictly on the provided document fragments. Provide only the most direct and concise answer. Do not include explanations, full sentences, or additional context.

The following are given document fragments.
        
{docs_block}

''').strip()

        user_prompt = f"Question: {question}\nAnswer:"

        # 注意：生成阶段不提供助手答案
        input_ids = _build_messages_input(tokenizer, system_prompt, user_prompt, assistant_content=None, use_chat_template=use_chat_template)
        inputs.append(input_ids)
        gold_answers_list.append(list(answers))

        # if i == 0:
        #     print(f"[Generation Chat Input Example]\nSYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}")

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
    parser.add_argument('--dataset_path', type=str, default='/root/shared-nvme/RAG-llm/RAG/data/test.jsonl')
    parser.add_argument('--use_chat_template', action='store_true')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--max_docs', type=int, default=None)
    parser.add_argument('--sample_size', type=int, default=300, help='随机抽取的样本数')
    parser.add_argument('--sample_seed', type=int, default=2025, help='随机抽样种子')
    parser.add_argument('--alpha', type=float, default=15.0, help='干预强度系数')
    parser.add_argument('--head_dim', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=None, help='若不传，将从特征维度推断')
    parser.add_argument('--pf_gamma', type=float, default=1.0, help='可靠性因子幂次 γ，用于 (reliability^γ)')
    # 已保存的探针与top-k头、验证准确率
    parser.add_argument('--probes_path', type=str, required=True)
    parser.add_argument('--top_heads_path', type=str, default=None)
    parser.add_argument('--val_accs_path', type=str, required=True)
    parser.add_argument('--select_top_k', type=int, default=48)
    # 用于计算 proj_val_std 的激活（推荐使用 NQ 收集的激活）
    parser.add_argument('--tuning_headwise_path', type=str, default='./RAG/features/llama2_chat_7B_nq_head_wise.npy')
    parser.add_argument('--tuning_labels_path', type=str, default='./RAG/features/llama2_chat_7B_nq_labels.npy')
    # 输出路径（同时评估标准RAG与探针干预RAG）
    parser.add_argument('--save_answers_baseline_path', type=str, default='./RAG/results_dump/main/llama-2-7b-instruct/answer_dump/nq_gen_answers_baseline.jsonl')
    parser.add_argument('--save_summary_baseline_path', type=str, default='./RAG/results_dump/main/llama-2-7b-instruct/summary_dump/nq_gen_summary_baseline.json')
    parser.add_argument('--save_answers_intervene_path', type=str, default='./RAG/results_dump/main/llama-2-7b-instruct/answer_dump/nq_gen_answers_intervene.jsonl')
    parser.add_argument('--save_summary_intervene_path', type=str, default='./RAG/results_dump/main/llama-2-7b-instruct/summary_dump/nq_gen_summary_intervene.json')
    parser.add_argument('--max_new_tokens', type=int, default=256, help='生成最大新token数（贪心解码）')
    args = parser.parse_args()

    MODEL = HF_NAMES.get(args.model_name, None)
    if MODEL is None:
        raise ValueError(f"不支持的模型名: {args.model_name}")

    # 加载分词器与模型（Llama3 使用 fast tokenizer 更兼容 tokenizer.json）
    tokenizer = llama.LlamaTokenizerFast.from_pretrained(MODEL)
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

    with open(args.probes_path, 'rb') as f:
        probes = pickle.load(f)
    val_accs = np.load(args.val_accs_path)
    L_va, H_va = val_accs.shape
    if args.top_heads_path:
        with open(args.top_heads_path, 'rb') as f:
            top_heads = pickle.load(f)
    else:
        if args.select_top_k is None or args.select_top_k <= 0:
            top_heads = [(l, h) for l in range(L_va) for h in range(H_va)]
        else:
            scores_flat = val_accs.reshape(L_va * H_va)
            idxs = np.argsort(scores_flat)[::-1][:args.select_top_k]
            top_heads = [(i // H_va, i % H_va) for i in idxs]

    # 加载用于计算 proj_val_std 的 head-wise 激活，并 reshape 为 (B, L, H, D)
    tuning_headwise = np.load(args.tuning_headwise_path)
    B, L, HD = tuning_headwise.shape
    if args.num_heads is None:
        if HD % args.head_dim != 0:
            raise ValueError('无法从特征维推断 num_heads，请显式传入 --num_heads')
        num_heads = HD // args.head_dim
    else:
        num_heads = args.num_heads
    tuning_headwise = rearrange(tuning_headwise, 'b l (h d) -> b l h d', h=num_heads, d=args.head_dim)
    tuning_labels = np.load(args.tuning_labels_path)
    num_questions = B // 2
    separated_head, separated_labels, _ = get_separated_activations_nq(tuning_labels, tuning_headwise, num_questions)
    train_set_idxs = np.arange(num_questions)
    val_set_idxs = np.array([], dtype=int)
    com_directions = get_com_directions(L, num_heads, train_set_idxs, val_set_idxs, separated_head, separated_labels)

    # 构造干预字典（包含探针分数因子）
    probe_score_map = val_accs
    interventions = get_interventions_dict(
        top_heads,
        probes,
        tuning_headwise,
        num_heads,
        use_center_of_mass=True,
        use_random_dir=False,
        com_directions=com_directions,
        probe_score_map=probe_score_map,
    )

    # 干预函数：在最后一个 token 的 head 输出上加方向
    def lt_modulated_vector_add(head_output, layer_name, start_edit_location='lt'):
        head_output = rearrange(head_output, 'b s (h d) -> b s h d', h=num_heads)
        # 归一化起始位置：默认最后一个 token；若传入非整数/不可转换，则回退为 -1
        if isinstance(start_edit_location, int):
            start_idx = max(0, start_edit_location)
        elif isinstance(start_edit_location, torch.Tensor):
            try:
                start_idx = int(start_edit_location.item())
            except Exception:
                start_idx = -1
        elif isinstance(start_edit_location, str) and start_edit_location == 'lt':
            start_idx = -1
        else:
            start_idx = -1

        # 解析层索引（model.layers.{i}.self_attn.head_out）
        try:
            layer_idx = int(str(layer_name).split('.')[2])
        except Exception:
            layer_idx = None

        for head, direction, proj_val_std, probe_factor in interventions[layer_name]:
            direction_to_add = torch.tensor(direction).to(head_output.device)
            # 动态分数：sigmoid(w·x + b)
            try:
                if layer_idx is not None:
                    flat_idx = layer_idx * num_heads + head
                    clf = probes[flat_idx]
                    w = torch.tensor(clf.coef_.reshape(-1), dtype=direction_to_add.dtype).to(head_output.device)
                    b = float(getattr(clf, 'intercept_', [0.0])[0])
                    x_vec = head_output[:, -1, head, :]  # B x D（贪心通常 B=1）
                    logit = (x_vec @ w) + b
                    dynamic_score = torch.sigmoid(logit)  # B
                else:
                    dynamic_score = torch.tensor(0.0, dtype=direction_to_add.dtype, device=head_output.device)
            except Exception:
                dynamic_score = torch.tensor(0.0, dtype=direction_to_add.dtype, device=head_output.device)

            reliability = float(probe_factor)
            strength_base = args.alpha * proj_val_std * (reliability ** args.pf_gamma)
            # strength_base = args.alpha * proj_val_std
            if start_idx == -1:
                # head_output[:, -1, head, :] += (strength_base * (1.0 - dynamic_score)).unsqueeze(-1) * direction_to_add
                head_output[:, -1, head, :] += strength_base * direction_to_add
            else:
                # head_output[:, start_idx:, head, :] += (strength_base * (1.0 - dynamic_score)).unsqueeze(-1) * direction_to_add
                head_output[:, start_idx:, head, :] += strength_base * direction_to_add
        head_output = rearrange(head_output, 'b s h d -> b s (h d)')
        return head_output

    # 逐条生成并评估（标准RAG 与 探针干预RAG）
    os.makedirs(os.path.dirname(args.save_answers_baseline_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.save_summary_baseline_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.save_answers_intervene_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.save_summary_intervene_path), exist_ok=True)

    preds_baseline = []
    preds_intervene = []

    def _clean_answer_text(gen_str: str) -> str:
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

    layers_to_intervene = list(interventions.keys())
    with torch.no_grad():
        for input_ids in tqdm(inputs, total=len(inputs), desc='baseline_generate'):
            input_ids = input_ids.to(device)
            gen_tokens_base = model.generate(
                input_ids,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
            )[:, input_ids.shape[-1]:]
            gen_str_base = tokenizer.decode(gen_tokens_base[0], skip_special_tokens=True)
            preds_baseline.append(_clean_answer_text(gen_str_base))

        for input_ids in tqdm(inputs, total=len(inputs), desc='intervene_generate'):
            input_ids = input_ids.to(device)
            with TraceDict(model, layers_to_intervene, edit_output=lt_modulated_vector_add):
                gen_tokens_itv = model.generate(
                    input_ids,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id,
                )[:, input_ids.shape[-1]:]
            gen_str_itv = tokenizer.decode(gen_tokens_itv[0], skip_special_tokens=True)
            preds_intervene.append(_clean_answer_text(gen_str_itv))

    # 保存逐条预测
    with open(args.save_answers_baseline_path, 'w', encoding='utf-8') as f:
        for pred, golds in zip(preds_baseline, gold_answers_list):
            item = {"prediction": pred, "gold_answers": golds}
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    with open(args.save_answers_intervene_path, 'w', encoding='utf-8') as f:
        for pred, golds in zip(preds_intervene, gold_answers_list):
            item = {"prediction": pred, "gold_answers": golds}
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # 计算并保存 EM/F1
    em_b, f1_b = evaluate_nq_em_f1(preds_baseline, gold_answers_list)
    summary_b = {"EM": em_b, "F1": f1_b, "alpha": 0.0, "model_name": args.model_name, "intervention": "none"}
    with open(args.save_summary_baseline_path, 'w', encoding='utf-8') as f:
        json.dump(summary_b, f, ensure_ascii=False, indent=2)

    em_i, f1_i = evaluate_nq_em_f1(preds_intervene, gold_answers_list)
    summary_i = {"EM": em_i, "F1": f1_i, "alpha": args.alpha, "model_name": args.model_name, "intervention": "probe_top48_with_factor"}
    with open(args.save_summary_intervene_path, 'w', encoding='utf-8') as f:
        json.dump(summary_i, f, ensure_ascii=False, indent=2)

    print("Baseline Summary:", summary_b)
    print("Intervention Summary:", summary_i)


if __name__ == '__main__':
    main()