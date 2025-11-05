"""
基于 RAG-ACT 的 new_dataset.jsonl，将 honest_llama/legacy 的两折探针训练流程迁移到 RAG 场景。

功能概述：
- 读取 collect_activations.py 生成的激活与标签（每个问题两条样本：正确=1，错误=0）。
- 将样本按问题分组（pair_size=2），做两折交叉验证：每次使用一折作为测试，剩余为开发集，再从开发集中划分验证集。
- 对每个层、每个注意力头训练一个 Logistic 回归探针，使用验证集评估准确率，选出 top-k 头。
- 计算沿探针方向在“调优激活”上的投影标准差，用于后续干预的强度归一化。
- 将当前折的探针、选中头、分割索引持久化保存，供 run_intervention_experiments.py 使用。

说明：
- 本脚本不依赖 TruthfulQA，只使用 RAG-ACT/data/new_dataset.jsonl 生成的激活数据。
- 训练探针时仅使用最后一个 token 的 head_out 表示（与 collect_activations.py 保持一致）。

注释风格：尽量用中文、面向入门者，避免晦涩术语。
"""

import os
import sys
import json
import pickle as pkl
from typing import List, Tuple, Dict, Any

import numpy as np
from tqdm import tqdm
from einops import rearrange

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import argparse
import llama


# 与项目其它脚本保持一致的模型名称映射（可按需增补）
HF_NAMES = {
    'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf',
    'llama3_8B': '/root/autodl-tmp/RAG-llm/models/Llama-3.1-8B-Instruct',
    'llama_7B': 'yahma/llama-7b-hf',
}


def split_by_pairs(labels: List[int], pair_size: int = 2) -> Tuple[List[List[int]], np.ndarray]:
    """
    将扁平标签列表按固定的 pair_size（每个问题两条样本：正确与错误）分组。

    返回：
    - separated_labels: 每个问题一个小列表，长度=pair_size（通常是 [1, 0] 或 [0, 1]）
    - idxs_to_split_at: 用于在激活数组的第一个维度（批次维）上进行 split 的边界索引

    这样我们可以把 head_wise_activations（形如 (B, L, H, D) 的数组）按问题切分，
    每个问题得到形如 (pair_size, L, H, D) 的小块，用于训练与验证。
    """
    if len(labels) % pair_size != 0:
        raise ValueError(
            f"标签长度 {len(labels)} 不能被 pair_size={pair_size} 整除。"
            f"请确保 collect_activations.py 为每个样本生成了恰好 {pair_size} 条输入（正确+错误）。"
        )
    idxs_to_split_at = np.arange(pair_size, len(labels) + 1, pair_size)
    labels = list(labels)
    separated_labels: List[List[int]] = []
    for i in range(len(idxs_to_split_at)):
        if i == 0:
            separated_labels.append(labels[:idxs_to_split_at[i]])
        else:
            separated_labels.append(labels[idxs_to_split_at[i - 1]:idxs_to_split_at[i]])
    return separated_labels, idxs_to_split_at


def train_probes(
    seed: int,
    train_question_idxs: np.ndarray,
    val_question_idxs: np.ndarray,
    separated_head_wise_activations: List[np.ndarray],
    separated_labels: List[List[int]],
    num_layers: int,
    num_heads: int,
) -> Tuple[List[Any], np.ndarray]:
    """
    训练“每个层-每个注意力头”的 Logistic 回归探针，并在验证集上评估准确率。

    输入：
    - train_question_idxs/val_question_idxs：问题级索引（不是样本级），用于从分组后的激活中取数据。
    - separated_head_wise_activations：长度为“问题数”的列表，每项形如 (pair_size, L, H, D)。
    - separated_labels：长度为“问题数”的列表，每项形如 (pair_size,) 的标签数组。
    - num_layers/num_heads：从模型配置中读取的层数与每层头数。

    输出：
    - probes：长度为 (num_layers * num_heads) 的探针列表，对应各层各头的 LogisticRegression 对象。
    - head_accs：形状为 (num_layers * num_heads,) 的验证准确率数组。
    """
    rng = np.random.RandomState(seed)
    probes: List[Any] = []
    head_accs: List[float] = []

    for layer in tqdm(range(num_layers), desc='训练探针（逐层）'):
        for head in range(num_heads):
            # 构造训练集特征与标签：把选中问题的两个样本（正确/错误）拼接在一起
            X_train = np.concatenate(
                [separated_head_wise_activations[i][:, layer, head, :] for i in train_question_idxs], axis=0
            )
            y_train = np.concatenate(
                [np.array(separated_labels[i]) for i in train_question_idxs], axis=0
            )

            # 验证集
            X_val = np.concatenate(
                [separated_head_wise_activations[i][:, layer, head, :] for i in val_question_idxs], axis=0
            )
            y_val = np.concatenate(
                [np.array(separated_labels[i]) for i in val_question_idxs], axis=0
            )

            # 若验证样本过少，跳过（避免数值不稳定）
            if X_val.shape[0] < 4:
                probes.append(LogisticRegression(random_state=seed, max_iter=1000))
                head_accs.append(0.0)
                continue

            # 训练 Logistic 回归探针。注意：这里用默认 L2 正则，max_iter 设大以确保收敛。
            clf = LogisticRegression(random_state=seed, max_iter=1000)
            clf.fit(X_train, y_train)
            y_pred_val = clf.predict(X_val)
            acc = accuracy_score(y_val, y_pred_val)

            probes.append(clf)
            head_accs.append(acc)

    return probes, np.array(head_accs)


def flattened_idx_to_layer_head(flat_idx: int, num_heads: int) -> Tuple[int, int]:
    """把展平索引还原为 (layer, head)。"""
    return flat_idx // num_heads, flat_idx % num_heads


def build_interventions_from_probes(
    top_heads: List[Tuple[int, int]],
    probes: List[Any],
    tuning_activations: np.ndarray,
    num_heads: int,
) -> Dict[str, List[Tuple[int, np.ndarray, float, Any]]]:
    """
    将选中的头与对应探针转换为“干预字典”，用于 TraceDict 注入。

    返回的字典形式：
    {"model.layers.{layer}.self_attn.head_out": [(head, direction, proj_std, probe), ...]}

    - direction 使用 Logistic 回归的 coef_，并做 L2 归一化。
    - proj_std 为在“调优激活(tuning_activations)”上投影值的标准差，用于尺度对齐。
    - tuning_activations 期望形状为 (N, L, H, D)，通常取训练问题的所有样本以避免泄漏测试信息。
    """
    interventions: Dict[str, List[Tuple[int, np.ndarray, float, Any]]] = {}

    for layer, head in top_heads:
        key = f"model.layers.{layer}.self_attn.head_out"
        if key not in interventions:
            interventions[key] = []

        probe = probes[layer * num_heads + head]
        direction = probe.coef_.astype(np.float32).copy()
        # L2 归一化方向
        norm = np.linalg.norm(direction)
        if norm == 0:
            # 极端情况：若方向全零，避免除零
            direction = direction
        else:
            direction = direction / norm

        # 计算该头在“调优激活”上的投影标准差，用于后续干预强度的缩放
        head_acts = tuning_activations[:, layer, head, :]  # (N, D)
        proj_vals = head_acts @ direction.reshape(-1, 1)   # (N, 1)
        proj_std = float(np.std(proj_vals))

        interventions[key].append((head, direction.squeeze(), proj_std, probe))

    # 按 head 顺序排序，便于可读与复现实验
    for key in interventions.keys():
        interventions[key] = sorted(interventions[key], key=lambda x: x[0])

    return interventions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama3_8B', help='模型名称（与 HF_NAMES 映射键一致）')
    parser.add_argument('--pair_size', type=int, default=2, help='每个问题的样本数（正确+错误）')
    parser.add_argument('--num_fold', type=int, default=2, help='折数（默认 2 折）')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='从开发集划分验证集比例')
    parser.add_argument('--top_k', type=int, default=48, help='选择用于干预的注意力头数量（跨所有层）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()

    print('Running:\n{}\n'.format(' '.join(sys.argv)))
    print(args)

    # 读取模型配置以获得层数与每层头数（用于 reshape 与遍历）
    model_id = HF_NAMES.get(args.model_name, None)
    if model_id is None:
        raise ValueError(f"不支持的模型名: {args.model_name}")
    cfg = llama.LlamaConfig.from_pretrained(model_id)
    num_layers = cfg.num_hidden_layers
    num_heads = cfg.num_attention_heads
    hidden_size = cfg.hidden_size
    head_dim = hidden_size // num_heads

    # 读取 collect_activations.py 输出的激活与标签
    head_wise_activations = pkl.load(open(f'./activations/{args.model_name}_head_wise.pkl', 'rb'))
    labels = pkl.load(open(f'./activations/{args.model_name}_labels.pkl', 'rb'))

    # 统一为 numpy，并 reshape 为 (B, L, H, D)
    head_wise_activations = np.array(head_wise_activations)  # (B, L, H*D)
    head_wise_activations = rearrange(head_wise_activations, 'b l (h d) -> b l h d', h=num_heads)

    # 按问题分组（每个问题 pair_size 条样本）
    separated_labels, idxs_to_split_at = split_by_pairs(labels, pair_size=args.pair_size)
    separated_head_wise_activations = np.split(head_wise_activations, idxs_to_split_at)
    # 现在每个元素形如 (pair_size, L, H, D)

    num_questions = len(separated_head_wise_activations)
    question_idxs = np.arange(num_questions)
    fold_blocks = np.array_split(question_idxs, args.num_fold)

    # 创建输出目录
    os.makedirs('./probes', exist_ok=True)
    os.makedirs('./splits', exist_ok=True)

    # 逐折训练与保存
    for i in range(args.num_fold):
        print(f"运行第 {i} 折")

        test_question_idxs = fold_blocks[i]
        dev_question_idxs = np.concatenate([fold_blocks[j] for j in range(args.num_fold) if j != i], axis=0)

        # 从开发集中划分验证集
        rng = np.random.RandomState(args.seed)
        n_dev = len(dev_question_idxs)
        n_val = int(max(1, np.floor(n_dev * args.val_ratio)))
        perm = rng.permutation(n_dev)
        val_question_idxs = dev_question_idxs[perm[:n_val]]
        train_question_idxs = dev_question_idxs[perm[n_val:]]

        # 保存分割索引，便于后续实验脚本复用
        split_obj = {
            'fold': i,
            'train_question_idxs': train_question_idxs.tolist(),
            'val_question_idxs': val_question_idxs.tolist(),
            'test_question_idxs': test_question_idxs.tolist(),
            'pair_size': args.pair_size,
        }
        with open(f'./splits/fold_{i}_seed_{args.seed}.json', 'w', encoding='utf-8') as f:
            json.dump(split_obj, f, ensure_ascii=False, indent=2)

        # 训练所有层-头的探针
        probes, head_accs = train_probes(
            seed=args.seed,
            train_question_idxs=train_question_idxs,
            val_question_idxs=val_question_idxs,
            separated_head_wise_activations=separated_head_wise_activations,
            separated_labels=separated_labels,
            num_layers=num_layers,
            num_heads=num_heads,
        )

        # 选取 top-k 头（过滤掉准确率为 0 的头）
        flat_order = np.argsort(head_accs)[::-1]
        flat_order = [idx for idx in flat_order if head_accs[idx] > 0.0]
        if len(flat_order) < args.top_k:
            print('警告：可用的有效头数量不足，实际选取数量为：', len(flat_order))
        chosen = flat_order[:args.top_k]
        top_heads: List[Tuple[int, int]] = [
            flattened_idx_to_layer_head(idx, num_heads) for idx in chosen
        ]
        print('选中的注意力头（layer, head）：', sorted(top_heads))

        # 计算“调优激活”：这里使用训练集问题的所有样本，避免信息泄露到测试集
        tuning_acts = np.concatenate(
            [separated_head_wise_activations[i] for i in train_question_idxs], axis=0
        )  # (N_train_samples, L, H, D)

        # 构造干预用字典（包含方向和投影标准差），并保存完整对象用于后续加载
        interventions = build_interventions_from_probes(
            top_heads=top_heads,
            probes=probes,
            tuning_activations=tuning_acts,
            num_heads=num_heads,
        )

        # 保存结果：探针、选中头、验证准确率、分割信息等
        dump_obj = {
            'model_name': args.model_name,
            'seed': args.seed,
            'fold': i,
            'num_layers': num_layers,
            'num_heads': num_heads,
            'head_dim': head_dim,
            'top_k': args.top_k,
            'head_accs': head_accs.tolist(),
            'top_heads': top_heads,
            'interventions': interventions,  # 直接保存，后续实验脚本可直接使用
        }
        out_path = f'./probes/{args.model_name}_seed_{args.seed}_top_{args.top_k}_fold_{i}.pkl'
        with open(out_path, 'wb') as f:
            pkl.dump(dump_obj, f)
        print('探针与干预参数已保存：', out_path)


if __name__ == '__main__':
    main()