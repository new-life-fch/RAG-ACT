# -*- coding: utf-8 -*-

"""
从 train.jsonl 中提取剩余的 1000 行作为测试集，并为每条样本保留三条检索片段（参照 generate_dataset.py 的逻辑）。

支持两种模式：
- first：训练集使用前 N 行，测试集为其余行（与 generate_dataset.py 中按顺序处理前 3000 行一致）
- random：训练集用随机种子从全体中采样 N 行，测试集为补集

示例：
python RAG/utils/extract_remaining_testset.py \
    --input /root/shared-nvme/RAG-llm/RAG/data/train.jsonl \
    --output /root/shared-nvme/RAG-llm/RAG/data/test.jsonl \
    --train-num 3000 --method first --seed 2025

若训练集为随机采样 3000 行（种子 2025），则：
python RAG/utils/extract_remaining_testset.py \
    --input /root/shared-nvme/RAG-llm/RAG/data/train.jsonl \
    --output /root/shared-nvme/RAG-llm/RAG/data/test.jsonl \
    --train-num 3000 --method random --seed 2025
"""

import argparse
import json
import os
import random


def select_random_passages(positive_passages, negative_passages, random_seed=2025):
    """
    参照 RAG/utils/generate_dataset.py 的逻辑：
    从 positive_passages 的第二个片段和 negative_passages 的前两个片段中随机选择 2 个，
    加上 positive_passages 的第一个片段，总共 3 个片段。
    """
    # 在函数内部设定随机种子，保证确定性（与原脚本一致）
    random.seed(random_seed)

    # 如果正向为空，退化为仅从负向选择最多 3 个
    if not positive_passages:
        candidates = negative_passages[:3] if isinstance(negative_passages, list) else []
        return candidates

    # 获取 positive_passages 的第一个片段（必选）
    first_positive = positive_passages[0]

    # 候选片段：positive 的第二个 + negative 的前两个
    candidates = []
    if isinstance(positive_passages, list) and len(positive_passages) > 1:
        candidates.append(positive_passages[1])
    if isinstance(negative_passages, list) and len(negative_passages) >= 2:
        candidates.extend(negative_passages[:2])

    # 从候选片段中随机选择 2 个（若不足则按可用数量）
    selected_candidates = random.sample(candidates, min(2, len(candidates))) if candidates else []

    # 组合结果：第一个 positive + 随机选择的 2 个
    result = [first_positive] + selected_candidates
    return result


def extract_testset(input_file: str, output_file: str, train_num: int, method: str = "first", seed: int = 2025) -> None:
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件不存在：{input_file}")

    # 读取所有行（JSONL）
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total = len(lines)
    if train_num > total:
        raise ValueError(f"训练样本数 train_num={train_num} 大于总行数 total={total}")

    if method not in {"first", "random"}:
        raise ValueError("method 必须为 'first' 或 'random'")

    if method == "first":
        # 训练集是前 train_num 行，测试集是剩余行
        test_indices = list(range(train_num, total))
    else:
        # 使用相同的随机采样策略重建训练索引，然后取补集
        rng = random.Random(seed)
        train_indices = set(rng.sample(range(total), train_num))
        test_indices = [i for i in range(total) if i not in train_indices]

    # 解析并生成仅保留三条检索片段的测试集样本
    test_entries = []
    for i in test_indices:
        raw = lines[i]
        try:
            data = json.loads(raw.strip())
        except Exception:
            # 若该行无法解析为 JSON，则原样保留为文本字段
            test_entries.append({"raw": raw.strip()})
            continue

        positive_passages = data.get("positive_passages", [])
        negative_passages = data.get("negative_passages", [])
        retrieve_snippets = select_random_passages(positive_passages, negative_passages, random_seed=seed)

        # 组装输出条目：保留必要字段与三条检索片段
        entry = {
            "query_id": data.get("query_id"),
            "query": data.get("query"),
            "answers": data.get("answers"),
            "retrieve_snippets": retrieve_snippets,
        }
        test_entries.append(entry)

    # 写出测试集
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in test_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print("提取完成！")
    print(f"输入总行数: {total}")
    print(f"训练集行数: {train_num}")
    print(f"测试集行数: {len(test_entries)}")
    print(f"输出文件: {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 train.jsonl 提取剩余行生成测试集（保留三条检索片段）")
    parser.add_argument("--input", type=str, default="/root/shared-nvme/RAG-llm/RAG/data/train.jsonl", help="输入 train.jsonl 路径")
    parser.add_argument("--output", type=str, default="/root/shared-nvme/RAG-llm/RAG/data/test.jsonl", help="输出测试集路径")
    parser.add_argument("--train-num", type=int, default=3000, help="训练集行数（剩余行将作为测试集）")
    parser.add_argument("--method", type=str, choices=["first", "random"], default="first", help="训练集选取方式：first 或 random")
    parser.add_argument("--seed", type=int, default=2025, help="随机模式下的随机种子，同时用于检索片段选择")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_testset(
        input_file=args.input,
        output_file=args.output,
        train_num=args.train_num,
        method=args.method,
        seed=args.seed,
    )