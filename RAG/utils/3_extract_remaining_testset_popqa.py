# -*- coding: utf-8 -*-

"""
从输入JSONL中拆分训练集、验证集、测试集，并为每条样本保留5条检索片段（参照原逻辑）。

支持两种模式：
- first：按顺序拆分（前N行=训练集，接下来M行=验证集，再接下来K行=测试集）
- random：按随机种子采样（先采训练集，再从剩余采验证集，最后从剩余采测试集）

示例：
python RAG/utils/3_extract_remaining_testset_popqa.py \
    --input RAG/data/NQ/nq-dev-train.jsonl \
    --val-output RAG/data/NQ/val.jsonl \
    --test-output RAG/data/NQ/test.jsonl \
    --train-num 1500 --val-num 1000 --test-num 2000 --method random --seed 2025
"""

import argparse
import json
import os
import random


def select_random_passages(positive_passages, negative_passages, random_seed=2025):
    """
    从positive_passages和negative_passages中选择5个片段。
    其中必须包含positive_passages的第一个片段。
    剩余4个片段从剩余的positive_passages和所有negative_passages中随机选择，
    使得每个片段（无论来自positive还是negative）被选中的概率相同。
    
    :param positive_passages: 正向片段列表
    :param negative_passages: 负向片段列表
    :param random_seed: 随机种子
    :return: 选择的5个片段列表
    """
    random.seed(random_seed)
    
    # 获取positive_passages的第一个片段（必选）
    first_positive = positive_passages[0] if positive_passages else {}
    
    # 构建候选池：positive_passages的剩余部分 + 全部negative_passages
    pool = []
    if len(positive_passages) > 1:
        pool.extend(positive_passages[1:])
    pool.extend(negative_passages)
    
    # 从候选池中随机选择4个片段（如果不足4个则全选）
    num_to_select = min(4, len(pool))
    selected_from_pool = random.sample(pool, num_to_select) if pool else []
    
    # 组合结果：第一个positive + 随机选择的片段
    result = [first_positive] + selected_from_pool
    
    return result


def select_passages_with_noise(positive_passages, negative_passages, noise_count, total=5, random_seed=2025):
    """
    片段选择（按噪声数）：在总数为 total 的前提下，目标包含 noise_count 条消极片段。

    约束：
    - 当 noise_count < total 且存在积极片段时，必须包含积极片段的第一条；
    - 当 noise_count == total 时，不包含积极片段（若消极不足则尽量补齐）。

    选择顺序：
    1)（可选）加入积极首片段；
    2) 随机选取 noise_count 条消极（若不足则尽可能多）；
    3) 用剩余槽位随机补充积极（排除首片段）；
    4) 若仍不足，用剩余消极补齐；

    去重：所有抽取均为无放回抽样。
    """
    random.seed(random_seed)
    result = []

    # 是否必须加入积极首片段（仅当噪声数不是 total）
    include_positive_head = (noise_count < total) and bool(positive_passages)
    if include_positive_head:
        result.append(positive_passages[0])

    slots_remaining = max(0, total - len(result))
    pos_pool = positive_passages[1:] if len(positive_passages) > 1 else []
    neg_pool = list(negative_passages)

    # 先满足目标消极数量
    need_neg = min(noise_count, total)
    take_neg = min(need_neg, len(neg_pool), slots_remaining)
    neg_selected = random.sample(neg_pool, take_neg) if take_neg > 0 else []
    result.extend(neg_selected)
    # 更新池与剩余槽位
    neg_pool = [x for x in neg_pool if x not in neg_selected]
    slots_remaining = max(0, total - len(result))

    # 再尝试用积极补齐到 total
    if slots_remaining > 0 and len(pos_pool) > 0:
        take_pos = min(slots_remaining, len(pos_pool))
        pos_selected = random.sample(pos_pool, take_pos)
        result.extend(pos_selected)
        pos_pool = [x for x in pos_pool if x not in pos_selected]
        slots_remaining = max(0, total - len(result))

    # 若仍未满，再用消极补齐
    if slots_remaining > 0 and len(neg_pool) > 0:
        take_neg_left = min(slots_remaining, len(neg_pool))
        if take_neg_left > 0:
            result.extend(random.sample(neg_pool, take_neg_left))

    return result


def process_entries(lines, indices, seed):
    """通用处理函数：根据索引生成条目（保留检索片段）"""
    entries = []
    for i in indices:
        raw = lines[i]
        try:
            data = json.loads(raw.strip())
        except Exception:
            # 若该行无法解析为 JSON，则原样保留为文本字段
            entries.append({"raw": raw.strip()})
            continue

        positive_passages = data.get("positive_passages", [])
        negative_passages = data.get("negative_passages", [])
        retrieve_snippets = select_random_passages(positive_passages, negative_passages, random_seed=seed)

        # 组装输出条目：保留必要字段与检索片段
        entry = {
            "query_id": data.get("id"),
            "query": data.get("question"),
            "answers": data.get("answers"),
            "retrieve_snippets": retrieve_snippets,
        }
        entries.append(entry)
    return entries


def process_entries_with_noise(lines, indices, seed, noise_count):
    """
    基于噪声数量生成条目：
    - 每条条目选择 total=5 个片段，其中噪声（消极）目标为 noise_count；
    - 当 noise_count < 5 且存在积极片段时，强制包含积极首片段；
    - 当 noise_count == 5 时，不包含积极片段（若消极不足则尽量补齐到 5）。
    """
    entries = []
    for i in indices:
        raw = lines[i]
        try:
            data = json.loads(raw.strip())
        except Exception:
            entries.append({"raw": raw.strip()})
            continue
        positive_passages = data.get("positive_passages", [])
        negative_passages = data.get("negative_passages", [])
        retrieve_snippets = select_passages_with_noise(positive_passages, negative_passages, noise_count, total=5, random_seed=seed)
        entry = {
            "query_id": data.get("id"),
            "query": data.get("question"),
            "answers": data.get("answers"),
            "retrieve_snippets": retrieve_snippets,
        }
        entries.append(entry)
    return entries


def extract_val_test_set(
    input_file: str,
    val_output_file: str,
    test_output_file: str,
    train_num: int,
    val_num: int,
    test_num: int,
    method: str = "first",
    seed: int = 2025,
    noise_levels=None
) -> None:
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件不存在：{input_file}")

    # 读取所有行（JSONL）
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total = len(lines)
    total_needed = train_num + val_num + test_num
    if total_needed > total:
        raise ValueError(
            f"总需求行数（训练{train_num}+验证{val_num}+测试{test_num}={total_needed}）"
            f" 大于总行数（{total}）"
        )

    if method not in {"first", "random"}:
        raise ValueError("method 必须为 'first' 或 'random'")

    # 划分训练/验证/测试集索引
    if method == "first":
        # 按顺序拆分
        train_indices = list(range(train_num))
        val_indices = list(range(train_num, train_num + val_num))
        test_indices = list(range(train_num + val_num, total_needed))
    else:
        # 随机采样拆分（无重叠）
        rng = random.Random(seed)
        all_indices = list(range(total))
        
        # 第一步：采样训练集
        train_indices = rng.sample(all_indices, train_num)
        remaining_after_train = [i for i in all_indices if i not in train_indices]
        
        # 第二步：从剩余中采样验证集
        val_indices = rng.sample(remaining_after_train, val_num)
        remaining_after_val = [i for i in remaining_after_train if i not in val_indices]
        
        # 第三步：从剩余中采样测试集
        test_indices = rng.sample(remaining_after_val, test_num)

    val_entries = process_entries(lines, val_indices, seed)
    test_entries = process_entries(lines, test_indices, seed)

    # 写出验证集
    os.makedirs(os.path.dirname(val_output_file), exist_ok=True)
    with open(val_output_file, "w", encoding="utf-8") as f:
        for entry in val_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # 写出测试集
    os.makedirs(os.path.dirname(test_output_file), exist_ok=True)
    with open(test_output_file, "w", encoding="utf-8") as f:
        for entry in test_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # 打印统计信息
    print("拆分完成！")
    print(f"输入总行数: {total}")
    print(f"训练集行数: {train_num}")
    print(f"验证集行数: {len(val_entries)} (输出: {val_output_file})")
    print(f"测试集行数: {len(test_entries)} (输出: {test_output_file})")
    print(f"总使用行数: {train_num + len(val_entries) + len(test_entries)}")

    if noise_levels:
        for n in noise_levels:
            val_noise_entries = process_entries_with_noise(lines, val_indices, seed, n)
            test_noise_entries = process_entries_with_noise(lines, test_indices, seed, n)
            val_noise_path = val_output_file[:-6] + f"_noise{n}.jsonl" if val_output_file.endswith(".jsonl") else val_output_file + f"_noise{n}"
            test_noise_path = test_output_file[:-6] + f"_noise{n}.jsonl" if test_output_file.endswith(".jsonl") else test_output_file + f"_noise{n}"
            os.makedirs(os.path.dirname(val_noise_path), exist_ok=True)
            with open(val_noise_path, "w", encoding="utf-8") as f:
                for entry in val_noise_entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            os.makedirs(os.path.dirname(test_noise_path), exist_ok=True)
            with open(test_noise_path, "w", encoding="utf-8") as f:
                for entry in test_noise_entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            print(f"噪声{n} 验证集行数: {len(val_noise_entries)} (输出: {val_noise_path})")
            print(f"噪声{n} 测试集行数: {len(test_noise_entries)} (输出: {test_noise_path})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="拆分JSONL为训练/验证/测试集（保留检索片段）")
    parser.add_argument("--input", type=str, required=True, help="输入JSONL文件路径")
    parser.add_argument("--val-output", type=str, required=True, help="验证集输出路径")
    parser.add_argument("--test-output", type=str, required=True, help="测试集输出路径")
    parser.add_argument("--train-num", type=int, required=True, help="训练集行数")
    parser.add_argument("--val-num", type=int, required=True, help="验证集行数")
    parser.add_argument("--test-num", type=int, required=True, help="测试集行数")
    parser.add_argument("--method", type=str, choices=["first", "random"], default="first", 
                        help="数据集拆分方式：first（顺序）/ random（随机）")
    parser.add_argument("--seed", type=int, default=2025, help="随机种子（用于采样和检索片段选择）")
    parser.add_argument("--noise-levels", type=str, default="", help="逗号分隔的噪声负向片段数量列表，如 '1,2,3,4,5'；留空不生成")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    noise_levels = None
    if args.noise_levels:
        try:
            noise_levels = [int(x) for x in args.noise_levels.split(",") if x.strip() != ""]
            noise_levels = [n for n in noise_levels if 0 <= n <= 5]
        except Exception:
            noise_levels = None
    extract_val_test_set(
        input_file=args.input,
        val_output_file=args.val_output,
        test_output_file=args.test_output,
        train_num=args.train_num,
        val_num=args.val_num,
        test_num=args.test_num,
        method=args.method,
        seed=args.seed,
        noise_levels=noise_levels,
    )
