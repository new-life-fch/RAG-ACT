import os
import json
import argparse
import glob
from typing import List, Dict, Any

def main():
    parser = argparse.ArgumentParser(description='Aggregate hyperparameter search summaries into a CSV file.')
    parser.add_argument('--results_root', type=str, required=True, help='Root directory containing summary_dump folder or the summary files directly.')
    parser.add_argument('--output_name', type=str, default='final_summary.csv', help='Name of the output CSV file.')
    args = parser.parse_args()

    # Determine summary directory
    summary_dir = os.path.join(args.results_root, 'summary_dump')
    if not os.path.exists(summary_dir):
        # Fallback: check if results_root itself contains json files
        if glob.glob(os.path.join(args.results_root, '*_summary.json')):
            summary_dir = args.results_root
        else:
            print(f"Error: Could not find 'summary_dump' directory in {args.results_root} or json files in root.")
            return

    print(f"Scanning for summary files in: {summary_dir}")

    # Load baseline
    baseline_path = os.path.join(summary_dir, 'baseline_summary.json')
    if not os.path.exists(baseline_path):
        # Try finding a file with "baseline" in the name
        candidates = glob.glob(os.path.join(summary_dir, '*baseline*summary.json'))
        if candidates:
            baseline_path = candidates[0]
        else:
            print("Warning: baseline_summary.json not found. d_EM and d_F1 will be 0.0.")
            baseline_sum = {"EM": 0.0, "F1": 0.0}
            baseline_path = None

    if baseline_path:
        try:
            with open(baseline_path, 'r', encoding='utf-8') as f:
                baseline_sum = json.load(f)
            print(f"Loaded baseline from {os.path.basename(baseline_path)}: EM={baseline_sum.get('EM', 0):.4f}, F1={baseline_sum.get('F1', 0):.4f}")
        except Exception as e:
            print(f"Error loading baseline: {e}")
            baseline_sum = {"EM": 0.0, "F1": 0.0}

    # Scan all summary files
    summary_files = glob.glob(os.path.join(summary_dir, '*_summary.json'))
    rows = []

    for fpath in summary_files:
        filename = os.path.basename(fpath)
        # Skip baseline file in the main loop
        if baseline_path and os.path.abspath(fpath) == os.path.abspath(baseline_path):
            continue
        
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Skipping {filename}: {e}")
            continue

        # Extract fields
        # Typical fields: "EM", "F1", "alpha", "intervention", "use_probe_factor", "num_heads_selected", "timed_out"
        
        # Filter out timed out or incomplete runs if necessary, but user might want partials.
        # The main script only saves if not timed out, but let's check flag.
        if data.get("timed_out", False):
            continue

        sel_name = data.get("intervention", "unknown")
        # Try to get num_heads from data, or parse from sel_name (e.g., topk_64_by_score)
        num_heads = data.get("num_heads_selected", 0)
        
        alpha = data.get("alpha", 0.0)
        use_pf = data.get("use_probe_factor", False)
        
        em = data.get("EM", 0.0)
        f1 = data.get("F1", 0.0)
        
        d_em = em - baseline_sum.get("EM", 0.0)
        d_f1 = f1 - baseline_sum.get("F1", 0.0)
        
        # Row format: selection,num_heads,alpha,use_probe_factor,EM,F1,d_EM,d_F1
        rows.append({
            "selection": sel_name,
            "num_heads": num_heads,
            "alpha": float(alpha),
            "use_probe_factor": int(use_pf),
            "EM": em,
            "F1": f1,
            "d_EM": d_em,
            "d_F1": d_f1
        })

    # Sort by F1 descending
    rows_sorted = sorted(rows, key=lambda x: x["F1"], reverse=True)

    # Write CSV
    output_path = os.path.join(args.results_root, args.output_name)
    headers = ['selection', 'num_heads', 'alpha', 'use_probe_factor', 'EM', 'F1', 'd_EM', 'd_F1']
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(','.join(headers) + '\n')
        for r in rows_sorted:
            line = [
                str(r["selection"]),
                str(r["num_heads"]),
                str(r["alpha"]),
                str(r["use_probe_factor"]),
                f"{r['EM']:.4f}",
                f"{r['F1']:.4f}",
                f"{r['d_EM']:.4f}",
                f"{r['d_F1']:.4f}"
            ]
            f.write(','.join(line) + '\n')

    print(f"Aggregated {len(rows_sorted)} results to {output_path}")

if __name__ == "__main__":
    main()
