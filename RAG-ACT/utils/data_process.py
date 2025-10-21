# -*- coding: utf-8 -*-

import random

def extract_random_jsonl_lines(input_file, output_file, num_lines, random_seed=None):
    """
    从一个 .jsonl 文件中随机提取 N 行到另一个文件。

    :param input_file: 输入的 .jsonl 文件路径
    :param output_file: 输出的 .jsonl 文件路径
    :param num_lines: 要提取的行数
    :param random_seed: 随机种子，用于确保结果可重现
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    try:
        # 首先读取所有行
        with open(input_file, 'r', encoding='utf-8') as infile:
            all_lines = infile.readlines()
        
        total_lines = len(all_lines)
        print(f"输入文件总行数: {total_lines}")
        
        # 检查要提取的行数是否超过总行数
        if num_lines > total_lines:
            print(f"警告：要提取的行数 ({num_lines}) 超过了文件总行数 ({total_lines})，将提取所有行")
            num_lines = total_lines
        
        # 随机选择行索引
        selected_indices = random.sample(range(total_lines), num_lines)
        selected_indices.sort()  # 排序以便按原始顺序输出
        
        # 写入选中的行
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for idx in selected_indices:
                outfile.write(all_lines[idx])
        
        print(f"成功随机提取了 {num_lines} 行到 '{output_file}'")
        print(f"使用随机种子: {random_seed}")

    except FileNotFoundError:
        print(f"错误：找不到输入文件 '{input_file}'")
    except Exception as e:
        print(f"处理文件时发生错误: {e}")

# --- 主程序 ---
if __name__ == "__main__":
    # 定义输入和输出文件名
    input_filename = './RAG-ACT/data/nq-dev_deduplicated.jsonl'  # 原始文件名
    output_filename = './RAG-ACT/data/train.jsonl' # 文件的名称   
    lines_to_extract = 3000          # 要提取的行数
    random_seed = 2025               # 随机种子

    # 调用函数执行任务
    extract_random_jsonl_lines(input_filename, output_filename, lines_to_extract, random_seed)


