#!/usr/bin/env python3
import argparse
import os
import sys
import pickle
import csv
from typing import List, Tuple, Union

import numpy as np


def load_val_accs(path: str) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 1:
        # Try to infer square matrix if flattened
        n = int(np.sqrt(arr.size))
        if n * n == arr.size:
            arr = arr.reshape(n, n)
        else:
            raise ValueError(f"Scores array must be 2D (layers x heads). Got shape {arr.shape}.")
    if arr.ndim != 2:
        raise ValueError(f"Scores array must be 2D (layers x heads). Got shape {arr.shape}.")
    return arr


def load_top_heads(path: str) -> List[Union[int, Tuple[int, int]]]:
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and 'top_heads' in obj:
        obj = obj['top_heads']
    if not isinstance(obj, (list, tuple)):
        raise ValueError("top_heads file must contain a list/tuple of indices or (layer, head) pairs.")
    return list(obj)


def index_to_layer_head(idx: Union[int, Tuple[int, int], List[int]], n_layers: int, n_heads: int) -> Tuple[int, int]:
    if isinstance(idx, (tuple, list)):
        if len(idx) != 2:
            raise ValueError(f"Invalid pair index: {idx}")
        l, h = int(idx[0]), int(idx[1])
        return l, h
    # assume flattened index
    i = int(idx)
    l = i // n_heads
    h = i % n_heads
    return l, h


def write_csv(rows: List[Tuple[int, int, float]], out_csv: str) -> None:
    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['layer', 'head', 'val_acc'])
        for l, h, acc in rows:
            w.writerow([l, h, f"{acc:.6f}"])


def main():
    parser = argparse.ArgumentParser(description="Inspect trained probe scores and optionally export CSV.")
    # Positional argument to match the request: probe score file path
    parser.add_argument('scores', help="Path to probe scores file (e.g., val_accs.npy)")
    parser.add_argument('--top-heads', dest='top_heads_path', default=None,
                        help="Optional path to top heads indices (pickle). Supports ints or (layer, head) pairs.")
    parser.add_argument('--out-csv', dest='out_csv', default=None,
                        help="Optional output CSV path. If omitted, a default under results/probes/ is used.")

    args = parser.parse_args()

    try:
        val_accs = load_val_accs(args.scores)
    except Exception as e:
        print(f"[Error] Failed to load scores from '{args.scores}': {e}", file=sys.stderr)
        sys.exit(1)

    n_layers, n_heads = val_accs.shape
    print(f"Loaded val_accs with shape: ({n_layers}, {n_heads})")

    # If top-heads provided, compute for those; else compute for all heads
    selected_rows: List[Tuple[int, int, float]] = []
    if args.top_heads_path:
        try:
            top_heads = load_top_heads(args.top_heads_path)
        except Exception as e:
            print(f"[Error] Failed to load top heads from '{args.top_heads_path}': {e}", file=sys.stderr)
            sys.exit(1)
        print(f"Loaded top heads count: {len(top_heads)}")

        for idx in top_heads:
            l, h = index_to_layer_head(idx, n_layers, n_heads)
            if not (0 <= l < n_layers and 0 <= h < n_heads):
                print(f"[Warn] Skipping out-of-range head index: {(l, h)}")
                continue
            acc = float(val_accs[l, h])
            selected_rows.append((l, h, acc))
    else:
        # Use all heads
        for l in range(n_layers):
            for h in range(n_heads):
                selected_rows.append((l, h, float(val_accs[l, h])))

    if not selected_rows:
        print("[Error] No valid heads to report.", file=sys.stderr)
        sys.exit(1)

    # Original order view
    print("\nOriginal order view:")
    for l, h, acc in selected_rows:
        print(f"Layer {l:02d} Head {h:02d} -> val_acc: {acc:.6f}")

    # Sorted view
    sorted_rows = sorted(selected_rows, key=lambda x: x[2], reverse=True)
    print("\nSorted by val_acc (desc):")
    for l, h, acc in sorted_rows:
        print(f"Layer {l:02d} Head {h:02d} -> val_acc: {acc:.6f}")

    # Summary stats
    accs = np.array([r[2] for r in selected_rows], dtype=np.float64)
    print("\nSummary:")
    print(f"Count: {accs.size}")
    print(f"Mean: {accs.mean():.6f}")
    print(f"Median: {np.median(accs):.6f}")
    print(f"Min: {accs.min():.6f}")
    print(f"Max: {accs.max():.6f}")

    # Decide CSV path
    out_csv = args.out_csv
    if out_csv is None:
        base_dir = os.path.join('results', 'probes')
        os.makedirs(base_dir, exist_ok=True)
        if args.top_heads_path:
            out_csv = os.path.join(base_dir, 'top_heads_val_accs.csv')
        else:
            out_csv = os.path.join(base_dir, 'all_heads_val_accs.csv')

    try:
        write_csv(sorted_rows, out_csv)
        print(f"\nSaved CSV to: {out_csv}")
    except Exception as e:
        print(f"[Error] Failed to write CSV to '{out_csv}': {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()