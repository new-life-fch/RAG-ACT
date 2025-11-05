import argparse
import sys
import os
import numpy as np
import pickle as pkl
from einops import rearrange
import llama

"""
Generate head-wise directions per question by contrasting correct vs. wrong answers.

This version is adapted to RAG-ACT/data/new_dataset.jsonl and the new
collect_activations.py outputs (two prompts per question: correct label=1,
wrong label=0). It removes TruthfulQA dependencies and computes per-question
splits directly from labels assuming a fixed pair size.
"""


HF_NAMES = {
    'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf',
    'llama3_8B': '/root/autodl-tmp/RAG-llm/models/Llama-3.1-8B-Instruct',
    'llama_7B': 'yahma/llama-7b-hf',
}


def split_by_pairs(labels, pair_size=2):
    """Split flat label list into per-question groups of size `pair_size`.

    Returns separated_labels list and idxs_to_split_at boundaries used to split
    activations along batch dimension.
    """
    if len(labels) % pair_size != 0:
        raise ValueError(f"Labels length {len(labels)} not divisible by pair_size={pair_size}.\n"
                         f"Ensure collect_activations.py produced exactly {pair_size} prompts per sample.")
    idxs_to_split_at = np.arange(pair_size, len(labels) + 1, pair_size)
    labels = list(labels)
    separated_labels = []
    for i in range(len(idxs_to_split_at)):
        if i == 0:
            separated_labels.append(labels[:idxs_to_split_at[i]])
        else:
            separated_labels.append(labels[idxs_to_split_at[i-1]:idxs_to_split_at[i]])
    return separated_labels, idxs_to_split_at

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama_7B')
    parser.add_argument('--pair_size', type=int, default=2, help='每个问题的样本数（正确+错误）')
    args = parser.parse_args()
    print('Running:\n{}\n'.format(' '.join(sys.argv)))
    print(args)

    # Load activations and labels produced by collect_activations.py
    head_wise_activations = pkl.load(open(f'./activations/{args.model_name}_head_wise.pkl', 'rb'))
    labels = pkl.load(open(f'./activations/{args.model_name}_labels.pkl', 'rb'))

    # Infer model heads from config for reliable reshape
    model_id = HF_NAMES.get(args.model_name, None)
    if model_id is None:
        raise ValueError(f"Unsupported model_name: {args.model_name}")
    try:
        cfg = llama.LlamaConfig.from_pretrained(model_id)
        num_heads = cfg.num_attention_heads
        hidden_size = cfg.hidden_size
    except Exception:
        # Fallback: infer heads from last-dim assuming typical head_dim of 128
        # This is a best-effort; prefer using config.
        sample = np.array(head_wise_activations)[0]
        hidden_size = sample.shape[-1]
        num_heads = 32 if hidden_size % 32 == 0 else 64
    head_dim = hidden_size // num_heads

    # Reshape to separate heads: (B, L, H, D)
    head_wise_activations = np.array(head_wise_activations)
    head_wise_activations = rearrange(head_wise_activations, 'b l (h d) -> b l h d', h=num_heads)

    # Split per question based on fixed pair size
    separated_labels, idxs_to_split_at = split_by_pairs(labels, pair_size=args.pair_size)
    separated_head_wise_activations = np.split(head_wise_activations, idxs_to_split_at)

    # Generate directions for each question: mean(pos) - mean(neg) over batch axis
    head_wise_activation_directions = np.array([
        a[np.array(l) == 1].mean(axis=0) - a[np.array(l) == 0].mean(axis=0)
        for a, l in zip(separated_head_wise_activations, separated_labels)
    ])  # shape: (num_questions, L, H, D)

    # Ensure output directory exists
    os.makedirs('./directions', exist_ok=True)
    pkl.dump(head_wise_activation_directions, open(f'./directions/{args.model_name}_directions.pkl', 'wb'))


if __name__ == '__main__':
    main()