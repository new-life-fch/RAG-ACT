# -*- coding: utf-8 -*-

"""
将 JSONL 数据集中每条数据的检索片段（retrieve_snippets）随机打乱顺序。

默认处理以下两个文件并就地覆盖：
- /root/shared-nvme/RAG-llm/RAG/new_dataset.jsonl
- /root/shared-nvme/RAG-llm/RAG/data/test.jsonl

随机种子默认 2025，保证可复现。

使用示例：
python RAG/utils/shuffle_retrieve_snippets.py --seed 2025

处理自定义文件：
python RAG/utils/shuffle_retrieve_snippets.py \
  --inputs RAG/data/NQ/test_noise_test_noise0.jsonl RAG/data/NQ/test_noise_test_noise1.jsonl RAG/data/NQ/test_noise_test_noise2.jsonl RAG/data/NQ/test_noise_test_noise3.jsonl RAG/data/NQ/test_noise_test_noise4.jsonl RAG/data/NQ/test_noise_test_noise5.jsonl --seed 2025
"""

import argparse
import json
import os
import random
from typing import List


def shuffle_snippets(snippets: List, rng: random.Random) -> List:
    if not isinstance(snippets, list) or len(snippets) <= 1:
        return snippets
    idxs = list(range(len(snippets)))
    rng.shuffle(idxs)
    return [snippets[i] for i in idxs]


def process_file(path: str, seed: int) -> None:
    if not os.path.exists(path):
        print(f"文件不存在，跳过：{path}")
        return

    rng = random.Random(seed)
    total = 0
    touched = 0
    out_lines = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            line_strip = line.strip()
            if not line_strip:
                out_lines.append("\n")
                continue
            try:
                obj = json.loads(line_strip)
            except Exception:
                # 保留无法解析的原始行
                out_lines.append(line)
                continue

            snippets = obj.get("retrieve_snippets")
            if isinstance(snippets, list) and len(snippets) > 1:
                obj["retrieve_snippets"] = shuffle_snippets(snippets, rng)
                touched += 1

            out_lines.append(json.dumps(obj, ensure_ascii=False) + "\n")

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(out_lines)

    print(f"处理完成：{path}")
    print(f"总行数: {total}，已打乱行数: {touched}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="随机打乱 JSONL 中每条数据的 retrieve_snippets 顺序")
    parser.add_argument(
        "--inputs",
        type=str,
        nargs="*",
        default=[
            "/root/shared-nvme/RAG-llm/RAG/new_dataset.jsonl",
            "/root/shared-nvme/RAG-llm/RAG/data/test.jsonl",
        ],
        help="要处理的 JSONL 文件路径，默认处理 new_dataset.jsonl 与 test.jsonl",
    )
    parser.add_argument("--seed", type=int, default=2025, help="随机种子，用于打乱片段顺序")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for p in args.inputs:
        process_file(p, args.seed)