"""
在 RAG 场景中加载已保存的 top-k 探针，注入模型的注意力头输出进行干预，
并在 new_dataset.jsonl 上运行两类实验：

1) 选择题式概率评估（MC）：
   - 计算在给定检索片段的系统+用户提示下，模型对“正确答案”和“错误答案”的对数概率和；
   - 统计正确答案的总概率是否高于错误答案，作为 MC 准确率。

2) 自由生成评估（Gen）：
   - 在同样的系统+用户提示下令模型生成答案；
   - 统计生成的答案是否与提供的正确答案之一匹配（忽略大小写与首尾空白），作为 EM（Exact Match）。
   - 同时给出“未干预”和“干预后”的对比结果。

注入方式：
- 复用 honest_llama/legacy 中的 head_out 加性注入（lt_modulated_vector_add）：
  在指定层的指定头上，沿探针方向向量添加 alpha * proj_std * direction。
  proj_std 为调优激活上的投影标准差，用于尺度归一化。

使用说明：
- 先运行 RAG-ACT/train_save_probes.py 生成探针文件（默认保存至 ./probes/）。
- 再运行本脚本，指定 --probes_path 加载探针与干预参数，即可进行评估与生成实验。

尽量使用中文注释，帮助初学者理解实现细节。
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any, Tuple

import numpy as np
from tqdm import tqdm
import pickle as pkl

import torch
from einops import rearrange
from baukit import TraceDict

import llama


# 与训练脚本一致的模型名称映射
HF_NAMES = {
    'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf',
    'llama3_8B': '/root/autodl-tmp/RAG-llm/models/Llama-3.1-8B-Instruct',
    'llama_7B': 'yahma/llama-7b-hf',
}


def _load_nq_jsonl(path: str, max_samples: int = None) -> List[Dict[str, Any]]:
    """读取 RAG-ACT 数据集（new_dataset.jsonl）。"""
    entries = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            data = json.loads(line.strip())
            entries.append(data)
    return entries


def _build_messages_input(tokenizer, system_prompt: str, user_prompt: str, assistant_content: str = None, use_chat_template: bool = True, for_generation: bool = False):
    """
    构造聊天消息输入：
    - 优先使用 tokenizer.apply_chat_template(messages, add_generation_prompt=<for_generation>)，再分词。
    - 若不可用则退化为简洁拼接文本（system + user [+ assistant]）。

    返回 input_ids（torch.LongTensor，形状 [1, seq_len]）。
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if assistant_content is not None:
        messages.append({"role": "assistant", "content": assistant_content})

    if use_chat_template:
        try:
            templated_text = tokenizer.apply_chat_template(messages, add_generation_prompt=for_generation)
            return tokenizer(templated_text, return_tensors='pt').input_ids
        except Exception:
            pass

    # 回退：直接拼接文本
    parts = [
        f"[SYSTEM]\n{system_prompt}\n[/SYSTEM]",
        f"[USER]\n{user_prompt}\n[/USER]",
    ]
    if assistant_content is not None:
        parts.append(f"[ASSISTANT]\n{assistant_content}\n[/ASSISTANT]")
    text = "\n\n".join(parts)
    return tokenizer(text, return_tensors='pt').input_ids


def _normalize_text(s: str) -> str:
    """简单归一化文本：小写 + 去除首尾空白。"""
    return (s or "").strip().lower()


def lt_modulated_vector_add_factory(num_heads: int, interventions: Dict[str, List[Tuple[int, np.ndarray, float, Any]]], alpha: float):
    """
    返回一个闭包函数，用于 TraceDict 的 edit_output：
    - 把 head_output 形状从 (b, s, h*d) 变成 (b, s, h, d)，便于对指定头进行向量加性注入；
    - 在指定 head 上添加 alpha * proj_std * direction；
    - 支持 start_edit_location：若为 'lt' 则只在序列最后一个位置注入；若为整数则从该位置至序列末尾均注入。
    """
    def lt_modulated_vector_add(head_output, layer_name, start_edit_location='lt', **kwargs):
        # 将 (b, s, h*d) 变形为 (b, s, h, d)，便于对每个头分别操作
        head_output = rearrange(head_output, 'b s (h d) -> b s h d', h=num_heads)
        edits = interventions.get(layer_name, [])

        if len(edits) == 0:
            head_output = rearrange(head_output, 'b s h d -> b s (h d)')
            return head_output

        # 根据起始位置选择注入范围
        if start_edit_location == 'lt':
            slc = slice(-1, None)  # 仅最后一个 token
        else:
            try:
                pos = int(start_edit_location)
            except Exception:
                pos = 0
            slc = slice(pos, None)

        # 对每个指定头进行注入
        for head, direction, proj_std, _probe in edits:
            # 把 numpy 的方向向量转到当前设备张量
            direction_vec = torch.tensor(direction, dtype=head_output.dtype, device=head_output.device)
            head_output[:, slc, head, :] = head_output[:, slc, head, :] + alpha * float(proj_std) * direction_vec

        # 还原形状回 (b, s, h*d)
        head_output = rearrange(head_output, 'b s h d -> b s (h d)')
        return head_output

    return lt_modulated_vector_add


def compute_answer_logprob(
    model, tokenizer, system_prompt: str, user_prompt: str, answer: str,
    use_chat_template: bool, interventions: Dict[str, Any], intervention_fn, start_edit_location: int
) -> float:
    """
    计算在给定系统+用户提示下，模型对“助手=答案”这段文本的总对数概率。
    实现思路与 tqa_run_probs 类似：
    - 先编码 system+user（不含助手内容），得到起始位置 start_edit_location；
    - 再编码包含 assistant=answer 的完整输入；
    - 只对 answer 部分的 token 计算 logprob 累积和；
    - 注入在 start_edit_location 及之后的 token（保证仅影响答案部分）。
    """
    with torch.no_grad():
        # system+user，仅用于确定答案开始位置（token 边界）
        ids_ctx = _build_messages_input(tokenizer, system_prompt, user_prompt, assistant_content=None, use_chat_template=use_chat_template, for_generation=False)
        ids_full = _build_messages_input(tokenizer, system_prompt, user_prompt, assistant_content=answer, use_chat_template=use_chat_template, for_generation=False)

        input_ids_ctx = ids_ctx.to(model.device)
        input_ids_full = ids_full.to(model.device)

        # 需要拦截的层（干预字典的键）
        layers_to_intervene = list(interventions.keys())
        if len(layers_to_intervene) == 0:
            # 无干预直接前向
            outputs = model(input_ids_full)[0].squeeze(0)
        else:
            with TraceDict(model, layers_to_intervene, edit_output=lambda x, ln: intervention_fn(x, ln, start_edit_location=start_edit_location)) as _ret:
                outputs = model(input_ids_full)[0].squeeze(0)

        # 转 log softmax
        outputs = outputs.log_softmax(-1)

        # 只取答案部分的 token 段
        ctx_len = input_ids_ctx.shape[-1]
        # 对齐：每个位置的概率对应“预测下一 token”，因此从 ctx_len-1 开始，到倒数第二个（最后一个 token 没有下一 token）
        outputs_ans = outputs[ctx_len - 1: -1, :]
        answer_ids = input_ids_full[0, ctx_len:]

        # 获取每个答案 token 的 logprob 并累加
        log_probs = outputs_ans[range(outputs_ans.shape[0]), answer_ids]
        total_logprob = float(log_probs.sum().item())
        return total_logprob


def generate_answer(
    model, tokenizer, system_prompt: str, user_prompt: str, use_chat_template: bool,
    max_new_tokens: int, interventions: Dict[str, Any], intervention_fn
) -> str:
    """
    在 system+user 背景下进行自由生成，返回生成的答案文本（去除特殊符号）。
    干预在序列最后一个 token（'lt'）处进行，常见于生成场景。
    """
    with torch.no_grad():
        ids_ctx = _build_messages_input(tokenizer, system_prompt, user_prompt, assistant_content=None, use_chat_template=use_chat_template, for_generation=True)
        input_ids = ids_ctx.to(model.device)

        layers_to_intervene = list(interventions.keys())
        if len(layers_to_intervene) == 0:
            gen = model.generate(input_ids, top_k=1, max_new_tokens=max_new_tokens, num_return_sequences=1)
        else:
            with TraceDict(model, layers_to_intervene, edit_output=lambda x, ln: intervention_fn(x, ln, start_edit_location='lt')) as _ret:
                gen = model.generate(input_ids, top_k=1, max_new_tokens=max_new_tokens, num_return_sequences=1)

        gen_tokens = gen[0, input_ids.shape[-1]:]
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        return gen_text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama3_8B', help='模型名称（与 HF_NAMES 映射键一致）')
    parser.add_argument('--probes_path', type=str, required=True, help='训练脚本保存的探针文件路径（.pkl）')
    parser.add_argument('--dataset_path', type=str, default='/root/autodl-tmp/RAG-llm/RAG-ACT/data/new_dataset.jsonl', help='RAG 数据集路径')
    parser.add_argument('--max_samples', type=int, default=None, help='仅评估前 N 条数据')
    parser.add_argument('--max_docs', type=int, default=4, help='每条样本使用的检索片段数量上限')
    parser.add_argument('--use_chat_template', action='store_true', help='使用 tokenizer.apply_chat_template 构造系统+用户输入')
    parser.add_argument('--alpha', type=float, default=15.0, help='干预强度系数 alpha')
    parser.add_argument('--max_new_tokens', type=int, default=32, help='生成的最大新 token 数')
    parser.add_argument('--compare_baseline', action='store_true', help='同时运行未干预的基线对照')
    args = parser.parse_args()

    print('Running:\n{}\n'.format(' '.join(sys.argv)))
    print(args)

    # 加载探针与干预参数
    with open(args.probes_path, 'rb') as f:
        probe_obj = pkl.load(f)

    model_name = probe_obj.get('model_name', args.model_name)
    model_id = HF_NAMES.get(model_name, None)
    if model_id is None:
        raise ValueError(f"不支持的模型名: {model_name}")

    # 模型与分词器：Llama3 建议 bfloat16；其他保持 float16
    tokenizer = llama.LlamaTokenizer.from_pretrained(model_id)
    dtype = torch.bfloat16 if 'llama3' in model_name else torch.float16
    model = llama.LlamaForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True, torch_dtype=dtype, device_map='auto')
    device = model.device
    model.eval()

    num_layers = probe_obj['num_layers']
    num_heads = probe_obj['num_heads']
    interventions = probe_obj['interventions']  # {layer_name: [(head, direction, proj_std, probe)]}

    # 生成干预函数闭包
    intervention_fn = lt_modulated_vector_add_factory(num_heads=num_heads, interventions=interventions, alpha=args.alpha)

    # 加载数据集
    entries = _load_nq_jsonl(args.dataset_path, max_samples=args.max_samples)

    # MC 与 Gen 指标累计
    mc_total = 0
    mc_correct = 0
    em_total = 0
    em_correct_edit = 0
    em_correct_base = 0

    for i, ex in enumerate(tqdm(entries, desc='评估与生成')):
        # 字段完整性检查
        if not all(k in ex for k in ["query", "answers", "wrong_answer", "retrieve_snippets"]):
            raise ValueError(f"第 {i} 条记录缺少必要字段：需包含 query/answers/wrong_answer/retrieve_snippets")

        question = ex['query']
        answers = ex['answers']  # 正确答案列表，取第一个为主
        wrong_answer = ex['wrong_answer']
        snippets = ex['retrieve_snippets']

        if not isinstance(answers, list) or len(answers) == 0:
            raise ValueError(f"第 {i} 条记录的 answers 非列表或为空，无法评估")

        # 取最多 max_docs 个检索文本，构造系统提示
        docs_texts = []
        for j, snip in enumerate(snippets):
            if args.max_docs is not None and j >= args.max_docs:
                break
            text = snip.get('text', '')
            if isinstance(text, str) and len(text.strip()):
                docs_texts.append(text.strip())

        docs_block = "\n\n".join([f"Document {k+1}: {d}" for k, d in enumerate(docs_texts)])
        system_prompt = (
            "Answer the question based on the given document. "
            "Provide only the most direct and concise answer. Do not include explanations, full sentences, or additional context. "
            "Just give the key information that directly answers the question.\n\n"
            f"The following are given documents.\n\n{docs_block}"
        )
        user_prompt = f"Question: {question}\nAnswer:"

        # ========== 1) MC 评估（正确 vs 错误的 logprob） ==========
        mc_total += 1
        # 首先构造起始位置：system+user 的 token 长度
        ids_ctx = _build_messages_input(tokenizer, system_prompt, user_prompt, assistant_content=None, use_chat_template=args.use_chat_template, for_generation=False)
        start_edit_location = int(ids_ctx.shape[-1])

        correct_ans = answers[0]
        logprob_correct = compute_answer_logprob(model, tokenizer, system_prompt, user_prompt, correct_ans, args.use_chat_template, interventions, intervention_fn, start_edit_location)
        logprob_wrong = compute_answer_logprob(model, tokenizer, system_prompt, user_prompt, wrong_answer, args.use_chat_template, interventions, intervention_fn, start_edit_location)

        if logprob_correct > logprob_wrong:
            mc_correct += 1

        # ========== 2) 生成评估（EM） ==========
        em_total += 1
        # 干预生成
        gen_edit = generate_answer(model, tokenizer, system_prompt, user_prompt, args.use_chat_template, args.max_new_tokens, interventions, intervention_fn)
        gen_edit_norm = _normalize_text(gen_edit)
        answers_norm = set(_normalize_text(a) for a in answers)
        if gen_edit_norm in answers_norm:
            em_correct_edit += 1

        # 可选：基线生成对照
        if args.compare_baseline:
            # 使用空干预
            gen_base = generate_answer(model, tokenizer, system_prompt, user_prompt, args.use_chat_template, args.max_new_tokens, interventions={}, intervention_fn=lambda x, ln, **kw: x)
            gen_base_norm = _normalize_text(gen_base)
            if gen_base_norm in answers_norm:
                em_correct_base += 1

    # 汇总结果
    mc_acc = mc_correct / max(1, mc_total)
    em_acc_edit = em_correct_edit / max(1, em_total)
    res = {
        'model_name': model_name,
        'alpha': args.alpha,
        'mc_acc': mc_acc,
        'em_acc_edit': em_acc_edit,
        'n_samples': len(entries) if args.max_samples is None else args.max_samples,
    }
    if args.compare_baseline:
        res['em_acc_base'] = em_correct_base / max(1, em_total)

    # 打印并保存到 results_dump
    os.makedirs('./results_dump/summary_dump', exist_ok=True)
    out_json = f'./results_dump/summary_dump/{model_name}_alpha_{int(args.alpha)}_rag_eval.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print('评估结果：')
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()