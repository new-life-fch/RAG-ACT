import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize, BoundaryNorm


def plot_heatmap_from_csv(input_csv: str, output_path: str = None, cmap: str = "turbo", discrete: bool = False, bins: int = 10) -> str:
    df = pd.read_csv(input_csv)
    if not set(["layer", "head", "val_acc"]).issubset(df.columns):
        raise ValueError("CSV需包含列: layer, head, val_acc")
    df["layer"] = pd.to_numeric(df["layer"], errors="coerce")
    df["head"] = pd.to_numeric(df["head"], errors="coerce")
    df["val_acc"] = pd.to_numeric(df["val_acc"], errors="coerce")
    df = df.dropna(subset=["layer", "head", "val_acc"]).astype({"layer": int, "head": int})

    mat = df.pivot(index="layer", columns="head", values="val_acc")
    mat = mat.sort_index().sort_index(axis=1)
    vmin = float(np.nanmin(mat.values))
    vmax = float(np.nanmax(mat.values))

    sns.set()
    plt.figure(figsize=(12, 8))
    norm = Normalize(vmin=vmin, vmax=vmax)
    if discrete:
        bounds = np.linspace(vmin, vmax, bins + 1)
        norm = BoundaryNorm(boundaries=bounds, ncolors=256)
    ax = sns.heatmap(mat, cmap=cmap, linewidths=0.3, linecolor="white", cbar=True, norm=norm)
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_title("Validation Accuracy Heatmap")
    plt.tight_layout()

    if output_path is None:
        base = os.path.splitext(os.path.basename(input_csv))[0]
        output_path = os.path.join(os.path.dirname(input_csv), base + "_heatmap.png")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def main():
    parser = argparse.ArgumentParser(description="从CSV绘制并保存热力图")
    parser.add_argument("csv", help="输入CSV路径，包含列layer,head,val_acc")
    parser.add_argument("--output", "-o", default=None, help="输出图片路径，默认与CSV同目录")
    parser.add_argument("--cmap", default="turbo", help="颜色映射，例如turbo、viridis、Spectral")
    parser.add_argument("--discrete", action="store_true", help="使用离散颜色分段以增强颜色区分度")
    parser.add_argument("--bins", type=int, default=10, help="离散颜色分段数")
    args = parser.parse_args()

    out = plot_heatmap_from_csv(args.csv, args.output, cmap=args.cmap, discrete=args.discrete, bins=args.bins)
    print(f"已保存热力图: {out}")


if __name__ == "__main__":
    main()

