#!/usr/bin/env python3
"""
数据集去重脚本
根据docid去除nq-dev.jsonl中positive_passages和negative_passages的重复片段
"""

import json
import argparse
from typing import Dict, List, Any
from collections import defaultdict


def deduplicate_passages(passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    根据docid去除重复的passages
    
    Args:
        passages: 包含docid的passage列表
        
    Returns:
        去重后的passage列表
    """
    seen_docids = set()
    deduplicated = []
    
    for passage in passages:
        docid = passage.get('docid')
        if docid and docid not in seen_docids:
            seen_docids.add(docid)
            deduplicated.append(passage)
    
    return deduplicated


def process_dataset(input_file: str, output_file: str) -> None:
    """
    处理整个数据集，去除重复片段
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
    """
    total_lines = 0
    processed_lines = 0
    total_positive_before = 0
    total_positive_after = 0
    total_negative_before = 0
    total_negative_after = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            total_lines += 1
            
            try:
                data = json.loads(line.strip())
                
                # 处理positive_passages
                if 'positive_passages' in data:
                    original_positive = data['positive_passages']
                    total_positive_before += len(original_positive)
                    data['positive_passages'] = deduplicate_passages(original_positive)
                    total_positive_after += len(data['positive_passages'])
                
                # 处理negative_passages
                if 'negative_passages' in data:
                    original_negative = data['negative_passages']
                    total_negative_before += len(original_negative)
                    data['negative_passages'] = deduplicate_passages(original_negative)
                    total_negative_after += len(data['negative_passages'])
                
                # 写入处理后的数据
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                processed_lines += 1
                
            except json.JSONDecodeError as e:
                print(f"警告：第{total_lines}行JSON解析失败: {e}")
                continue
    
    # 打印统计信息
    print(f"处理完成！")
    print(f"总行数: {total_lines}")
    print(f"成功处理行数: {processed_lines}")
    print(f"Positive passages: {total_positive_before} -> {total_positive_after} (去除 {total_positive_before - total_positive_after} 个重复)")
    print(f"Negative passages: {total_negative_before} -> {total_negative_after} (去除 {total_negative_before - total_negative_after} 个重复)")


def analyze_duplicates(input_file: str) -> None:
    """
    分析数据集中每个条目内部的重复情况
    
    Args:
        input_file: 输入文件路径
    """
    total_lines = 0
    entries_with_positive_duplicates = 0
    entries_with_negative_duplicates = 0
    total_positive_duplicates = 0
    total_negative_duplicates = 0
    max_positive_duplicates_in_entry = 0
    max_negative_duplicates_in_entry = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line_num, line in enumerate(infile, 1):
            total_lines += 1
            
            try:
                data = json.loads(line.strip())
                
                # 分析positive_passages中的重复
                if 'positive_passages' in data:
                    positive_docids = []
                    for passage in data['positive_passages']:
                        docid = passage.get('docid')
                        if docid:
                            positive_docids.append(docid)
                    
                    # 统计重复
                    unique_positive = len(set(positive_docids))
                    total_positive = len(positive_docids)
                    duplicates_in_entry = total_positive - unique_positive
                    
                    if duplicates_in_entry > 0:
                        entries_with_positive_duplicates += 1
                        total_positive_duplicates += duplicates_in_entry
                        max_positive_duplicates_in_entry = max(max_positive_duplicates_in_entry, duplicates_in_entry)
                        
                        # 显示前几个有重复的条目作为示例
                        if entries_with_positive_duplicates <= 3:
                            print(f"示例 - 第{line_num}行positive_passages有{duplicates_in_entry}个重复片段")
                
                # 分析negative_passages中的重复
                if 'negative_passages' in data:
                    negative_docids = []
                    for passage in data['negative_passages']:
                        docid = passage.get('docid')
                        if docid:
                            negative_docids.append(docid)
                    
                    # 统计重复
                    unique_negative = len(set(negative_docids))
                    total_negative = len(negative_docids)
                    duplicates_in_entry = total_negative - unique_negative
                    
                    if duplicates_in_entry > 0:
                        entries_with_negative_duplicates += 1
                        total_negative_duplicates += duplicates_in_entry
                        max_negative_duplicates_in_entry = max(max_negative_duplicates_in_entry, duplicates_in_entry)
                        
                        # 显示前几个有重复的条目作为示例
                        if entries_with_negative_duplicates <= 3:
                            print(f"示例 - 第{line_num}行negative_passages有{duplicates_in_entry}个重复片段")
                            
            except json.JSONDecodeError:
                continue
    
    print(f"\n数据集内部重复分析结果:")
    print(f"总条目数: {total_lines}")
    print(f"包含positive_passages重复的条目数: {entries_with_positive_duplicates}")
    print(f"包含negative_passages重复的条目数: {entries_with_negative_duplicates}")
    print(f"positive_passages总重复片段数: {total_positive_duplicates}")
    print(f"negative_passages总重复片段数: {total_negative_duplicates}")
    print(f"单个条目中positive_passages最多重复数: {max_positive_duplicates_in_entry}")
    print(f"单个条目中negative_passages最多重复数: {max_negative_duplicates_in_entry}")


def main():
    parser = argparse.ArgumentParser(description='数据集去重工具')
    parser.add_argument('--input', '-i', required=True, help='输入文件路径')
    parser.add_argument('--output', '-o', help='输出文件路径（如果不指定则为输入文件名_deduplicated.jsonl）')
    parser.add_argument('--analyze', '-a', action='store_true', help='仅分析重复情况，不进行去重')
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_duplicates(args.input)
    else:
        if not args.output:
            # 生成默认输出文件名
            input_parts = args.input.rsplit('.', 1)
            if len(input_parts) == 2:
                args.output = f"{input_parts[0]}_deduplicated.{input_parts[1]}"
            else:
                args.output = f"{args.input}_deduplicated"
        
        print(f"开始处理数据集...")
        print(f"输入文件: {args.input}")
        print(f"输出文件: {args.output}")
        
        process_dataset(args.input, args.output)


if __name__ == "__main__":
    main()