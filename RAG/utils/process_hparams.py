import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import os

# 设置中文字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def process_hparams(input_path):
    print(f"Reading file: {input_path}")
    if not os.path.exists(input_path):
        print(f"Error: File {input_path} not found.")
        return

    # 1. 读取数据
    df = pd.read_csv(input_path)
    
    # 2. 需求 1: 更改表示方法，列为alpha，行为selection，剩下为F1值，生成新的csv
    # 我们先按照 alpha 排序，保证列的顺序
    pivot_csv_df = df.pivot(index='selection', columns='alpha', values='F1')
    
    # 按照 selection 中的数字排序行，让结果更整齐
    def extract_num(s):
        match = re.search(r'(\d+)', str(s))
        return int(match.group(1)) if match else 0
    
    pivot_csv_df['temp_sort'] = pivot_csv_df.index.map(extract_num)
    pivot_csv_df = pivot_csv_df.sort_values('temp_sort').drop(columns=['temp_sort'])
    
    output_dir = os.path.dirname(input_path)
    csv_output_path = os.path.join(output_dir, "超参结果_重排.csv")
    pivot_csv_df.to_csv(csv_output_path)
    print(f"Saved pivoted CSV to: {csv_output_path}")

    # 3. 需求 2 & 3: 画热力图
    # 提取 selection 中的数字作为纵轴
    df['Numbers of Heads InterVened K'] = df['selection'].apply(extract_num)
    df['Intervention Strength'] = df['alpha']
    df['F1(%)'] = df['F1'] * 100
    
    # 构建热力图矩阵
    heatmap_df = df.pivot(index='Numbers of Heads InterVened K', columns='Intervention Strength', values='F1(%)')
    heatmap_df = heatmap_df.sort_index(ascending=True) # 纵轴数字从小到大
    heatmap_df = heatmap_df.sort_index(axis=1, ascending=True) # 横轴 alpha 从小到大

    # 绘图
    plt.figure(figsize=(12, 10))
    
    # 颜色要求：主体蓝色，脱离主体（通常是较低的值）趋近白色
    # 因为 F1 值大部分接近，我们使用 robust=True 自动调整颜色范围，或者根据数据动态调整
    # sns.heatmap 的 cmap="Blues" 是从白到蓝。
    # 为了让“颜色分明”，我们可以手动计算 vmin/vmax，或者使用 robust=True
    
    # 计算一些统计值来辅助颜色映射
    all_values = df['F1(%)'].values
    # 为了让“颜色分明”，我们手动设置 vmin 和 vmax
    # 我们可以将 vmin 设置为较低的分位数，使得大部分值在蓝色区域有明显的梯度
    vmin = all_values.min()
    vmax = all_values.max()
    
    # 绘图
    plt.figure(figsize=(12, 10))
    
    # 使用 robust=True 可以自动处理离群值，使主体颜色分明
    # 或者手动微调：让大部分值（主体）处于蓝色区域，极小值（脱离主体）趋近白色
    ax = sns.heatmap(
        heatmap_df, 
        annot=True, 
        fmt=".2f", 
        cmap="Blues", 
        linewidths=.5,
        robust=True, # 自动调整颜色范围以适应主体数据
        cbar_kws={'label': 'F1 (%)'}
    )
    
    plt.title('F1(%)', fontsize=15)
    plt.xlabel('Intervention Strength', fontsize=12)
    plt.ylabel('Numbers of Heads InterVened K', fontsize=12)
    
    img_output_path = os.path.join(output_dir, "hparam_heatmap.png")
    plt.savefig(img_output_path, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap to: {img_output_path}")

if __name__ == "__main__":
    input_file = "/root/shared-nvme/RAG-llm/RAG/results/hparam-search-popqa/超参结果.csv"
    process_hparams(input_file)
