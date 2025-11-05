import os
import torch
from tqdm import tqdm
import pickle
import argparse
import sys
import json
from typing import List, Tuple, Any

import llama
from utils import get_llama_activations_bau


def _format_nq_prompt_with_docs(question: str, docs: List[str], answer: str) -> str:
    docs_block = "\n\n".join([f"Document {i+1}: {d}" for i, d in enumerate(docs)])
    # 维持 Q:/A: 标记，以与既有流程兼容
    return f"Q: {question}\nDocuments:\n{docs_block}\nA: {answer}"


def _load_nq_jsonl(path: str, max_samples: int = None) -> List[Any]:
    entries = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            data = json.loads(line.strip())
            entries.append(data)
    return entries


def _build_messages_input(tokenizer, system_prompt: str, user_prompt: str, assistant_content: str = None, use_chat_template: bool = True):
    """
    根据系统/用户/助手消息构建输入。
    - 优先使用 tokenizer.apply_chat_template(messages, add_generation_prompt=False) 生成模板化文本，再分词。
    - 若不可用则退化为简洁的三段文本拼接（system + user + assistant）。

    注意：这里将助手回答作为 assistant 角色消息，以模拟真实对话（teacher forcing）。
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if assistant_content is not None:
        messages.append({"role": "assistant", "content": assistant_content})

    if use_chat_template:
        try:
            # 生成模板化文本，再交由 tokenizer 编码，兼容不同版本的 HF
            templated_text = tokenizer.apply_chat_template(messages, add_generation_prompt=False)
            return tokenizer(templated_text, return_tensors='pt').input_ids
        except Exception:
            pass

    # 回退：直接拼接 system + user + assistant 文本
    parts = [
        f"[SYSTEM]\n{system_prompt}\n[/SYSTEM]",
        f"[USER]\n{user_prompt}\n[/USER]",
    ]
    if assistant_content is not None:
        parts.append(f"[ASSISTANT]\n{assistant_content}\n[/ASSISTANT]")
    text = "\n\n".join(parts)
    return tokenizer(text, return_tensors='pt').input_ids


def tokenized_nq_with_docs_dual(
    jsonl_path: str,
    tokenizer,
    max_samples: int = None,
    max_docs: int = None,
    use_chat_template: bool = False,
) -> Tuple[List[torch.Tensor], List[int], List[str], List[List[str]]]:
    """
    读取 new_dataset.jsonl，为每条样本生成两种输入：
    - 问题 + 检索片段 + 正确答案（label=1）
    - 问题 + 检索片段 + 错误答案（label=0）

    返回与原 tokenized_tqa_all 一致的四元组 (prompts, labels, categories, tokens)。
    """
    entries = _load_nq_jsonl(jsonl_path, max_samples=max_samples)

    prompts: List[torch.Tensor] = []
    labels: List[int] = []
    categories: List[str] = []
    tokens: List[List[str]] = []

    for i, ex in enumerate(entries):
        # 字段完整性检查
        if not all(k in ex for k in ["query", "answers", "wrong_answer", "retrieve_snippets"]):
            raise ValueError(f"new_dataset.jsonl 第 {i} 条记录缺少必要字段，需包含 query/answers/wrong_answer/retrieve_snippets")

        question = ex["query"]
        answers = ex["answers"]
        wrong_answer = ex["wrong_answer"]
        snippets = ex["retrieve_snippets"]

        if not isinstance(answers, list) or len(answers) == 0:
            raise ValueError(f"第 {i} 条记录的 answers 非列表或为空，无法构造正确答案输入")

        # 取最多 max_docs 个片段文本
        docs_texts = []
        for j, snip in enumerate(snippets):
            if max_docs is not None and j >= max_docs:
                break
            text = snip.get("text", "")
            if isinstance(text, str) and len(text.strip()):
                docs_texts.append(text.strip())

        # 构造系统/用户提示词
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

        # 正确答案：作为助手消息（teacher forcing），以贴近真实对话流程（label=1）
        correct_answer = answers[0]
        user_prompt = f"Question: {question}\nAnswer:"
        correct_ids = _build_messages_input(tokenizer, system_prompt, user_prompt, correct_answer, use_chat_template)
        if i == 0:
            # 仅打印一次示例，避免刷屏
            print(f"[Correct Chat Input]\nSYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}\n\nASSISTANT:\n{correct_answer}")
        correct_tokens = tokenizer.convert_ids_to_tokens(correct_ids[0])
        prompts.append(correct_ids)
        labels.append(1)
        categories.append('NQ')
        tokens.append(correct_tokens)

        # 错误答案：同样作为助手消息（label=0）
        wrong_ids = _build_messages_input(tokenizer, system_prompt, user_prompt, wrong_answer, use_chat_template)
        if i == 0:
            print(f"[Wrong Chat Input]\nSYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}\n\nASSISTANT:\n{wrong_answer}")
        wrong_tokens = tokenizer.convert_ids_to_tokens(wrong_ids[0])
        prompts.append(wrong_ids)
        labels.append(0)
        categories.append('NQ')
        tokens.append(wrong_tokens)

    return prompts, labels, categories, tokens


def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama_7B')
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--dataset_path', type=str, default='/root/autodl-tmp/RAG-llm/RAG-ACT/data/new_dataset.jsonl')
    parser.add_argument('--max_samples', type=int, default=None, help='仅处理指定数量的样本')
    parser.add_argument('--max_docs', type=int, default=None, help='每条样本使用的检索片段数量上限')
    parser.add_argument('--use_chat_template', action='store_true', help='使用 tokenizer.apply_chat_template 构造系统+用户聊天输入')
    args = parser.parse_args()

    HF_NAMES = {
        'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf', 
        'llama3_8B': '/root/autodl-tmp/RAG-llm/models/Llama-3.1-8B-Instruct'
    }
    print('Running:\n{}\n'.format(' '.join(sys.argv)))
    print(args)

    MODEL = HF_NAMES.get(args.model_name, None)
    if MODEL is None:
        raise ValueError(f"不支持的模型名: {args.model_name}")

    tokenizer = llama.LlamaTokenizer.from_pretrained(MODEL)
    # Llama3 建议使用 bfloat16；其他模型保持 float16
    dtype = torch.bfloat16 if 'llama3' in args.model_name else torch.float16
    model = llama.LlamaForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, torch_dtype=dtype, device_map='auto')
    device = model.device

    # 加载并格式化 NQ 子集（包含正确/错误答案两种输入，带检索片段）
    print("Tokenizing prompts (NQ subset, with system+user chat, correct+wrong)")
    use_chat_template = args.use_chat_template
    prompts, labels, categories, tokens = tokenized_nq_with_docs_dual(
        args.dataset_path,
        tokenizer,
        max_samples=args.max_samples,
        max_docs=args.max_docs,
        use_chat_template=use_chat_template,
    )

    print("Getting activations")
    all_layer_wise_activations = []
    all_head_wise_activations = []
    for prompt, token in tqdm(zip(prompts, tokens), total=len(prompts)):
        # layer_wise_activations (33, 42, 4096) num_hidden_layers + last, seq_len, hidden_size
        # head_wise_activations (32, 42, 4096) num_hidden_layers, seq_len, hidden_size
        layer_wise_activations, head_wise_activations, _ = get_llama_activations_bau(model, prompt, device)
        all_layer_wise_activations.append(layer_wise_activations[:, -1, :])
        all_head_wise_activations.append(head_wise_activations[:, -1, :])

    # 确保输出目录存在
    os.makedirs('./activations', exist_ok=True)
    print("saving categories")
    pickle.dump(categories, open(f'./activations/{args.model_name}_categories.pkl', 'wb'))

    print("Saving labels")
    pickle.dump(labels, open(f'./activations/{args.model_name}_labels.pkl', 'wb'))
    
    print("Saving tokens")
    pickle.dump(tokens, open(f'./activations/{args.model_name}_tokens.pkl', 'wb'))

    print("Saving layer wise activations")
    pickle.dump(all_layer_wise_activations, open(f'./activations/{args.model_name}_layer_wise.pkl', 'wb'))
    
    print("Saving head wise activations")
    pickle.dump(all_head_wise_activations, open(f'./activations/{args.model_name}_head_wise.pkl', 'wb'))

    print("All saved successfully")

if __name__ == '__main__':
    main()