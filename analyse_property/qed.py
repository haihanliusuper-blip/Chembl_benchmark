# -*- coding: utf-8 -*-
import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)


def series_stats(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
            "q1": None,
            "q3": None,
        }
    return {
        "count": int(len(s)),
        "min": float(s.min()),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
        "q1": float(s.quantile(0.25)),
        "q3": float(s.quantile(0.75)),
    }


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def moving_average(y, window=5):
    if len(y) == 0:
        return y
    window = max(1, int(window))
    if window == 1:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def plot_qed_hist_with_curve(qed_values, out_png, bins=40, smooth_window=5):
    qed_values = pd.to_numeric(qed_values, errors="coerce").dropna().values
    if len(qed_values) == 0:
        return

    counts, bin_edges = np.histogram(qed_values, bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    smooth_counts = moving_average(counts, window=smooth_window)

    plt.figure(figsize=(8, 5.5), dpi=200)

    # 直方图
    plt.hist(qed_values, bins=bins, alpha=0.75, edgecolor="black", linewidth=0.5)

    # 平滑曲线
    plt.plot(bin_centers, smooth_counts, linewidth=2.0)

    plt.title("QED Distribution of Unique SMILES")
    plt.xlabel("QED")
    plt.ylabel("Count")
    plt.xlim(left=0)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze QED distribution based on unique SMILES")
    parser.add_argument(
        "--input_csv",
        type=str,
        default="all_valid_targets_compound_properties.csv",
        help="Input CSV file path"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="QED分析_唯一SMILES",
        help="Output directory"
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=40,
        help="Number of histogram bins"
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=5,
        help="Moving average window for smooth curve"
    )
    args = parser.parse_args()

    safe_mkdir(args.outdir)

    if not os.path.isfile(args.input_csv):
        print(f"[ERR] 找不到输入文件: {args.input_csv}")
        sys.exit(1)

    df = pd.read_csv(args.input_csv)

    required_cols = ["smiles", "qed"]
    for col in required_cols:
        if col not in df.columns:
            print(f"[ERR] 输入文件缺少必要列: {col}")
            sys.exit(2)

    # 只保留 smiles 和 qed，按 smiles 去重
    df_unique = df[["smiles", "qed"]].copy()
    df_unique["qed"] = pd.to_numeric(df_unique["qed"], errors="coerce")
    df_unique = df_unique.dropna(subset=["smiles", "qed"])
    df_unique = df_unique.drop_duplicates(subset=["smiles"]).reset_index(drop=True)

    qed_stats = series_stats(df_unique["qed"])

    # 分箱区间统计
    qed_bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    qed_labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    df_unique["qed_bin"] = pd.cut(
        df_unique["qed"],
        bins=qed_bins,
        labels=qed_labels,
        include_lowest=True,
        right=True
    )
    bin_counts = df_unique["qed_bin"].value_counts().reindex(qed_labels, fill_value=0)

    summary = {
        "n_unique_smiles": int(len(df_unique)),
        "qed_stats": qed_stats,
        "qed_bin_distribution": {str(k): int(v) for k, v in bin_counts.items()}
    }

    save_json(summary, os.path.join(args.outdir, "qed_unique_summary.json"))

    # 保存去重后的 QED 表
    df_unique.to_csv(
        os.path.join(args.outdir, "unique_smiles_qed.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # 作图
    plot_qed_hist_with_curve(
        qed_values=df_unique["qed"],
        out_png=os.path.join(args.outdir, "qed_hist_with_curve_unique.png"),
        bins=args.bins,
        smooth_window=args.smooth_window
    )

    # 文本摘要
    lines = []
    lines.append("QED analysis based on unique SMILES")
    lines.append("=" * 50)
    lines.append(f"Number of unique SMILES: {len(df_unique)}")
    lines.append("")
    lines.append("QED statistics")
    lines.append(f"Count  : {qed_stats['count']}")
    lines.append(f"Min    : {qed_stats['min']:.6f}")
    lines.append(f"Max    : {qed_stats['max']:.6f}")
    lines.append(f"Mean   : {qed_stats['mean']:.6f}")
    lines.append(f"Median : {qed_stats['median']:.6f}")
    lines.append(f"Std    : {qed_stats['std']:.6f}")
    lines.append(f"Q1     : {qed_stats['q1']:.6f}")
    lines.append(f"Q3     : {qed_stats['q3']:.6f}")
    lines.append("")
    lines.append("QED bin distribution")
    for k, v in bin_counts.items():
        ratio = v / len(df_unique) if len(df_unique) > 0 else 0
        lines.append(f"{k}: {int(v)} ({ratio:.2%})")

    with open(os.path.join(args.outdir, "qed_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # print 结果
    print("\n===== QED unique SMILES analysis =====")
    print(f"Unique SMILES count: {len(df_unique)}")
    print(f"QED mean   : {qed_stats['mean']:.6f}")
    print(f"QED median : {qed_stats['median']:.6f}")
    print(f"QED std    : {qed_stats['std']:.6f}")
    print(f"QED min    : {qed_stats['min']:.6f}")
    print(f"QED max    : {qed_stats['max']:.6f}")
    print(f"QED Q1     : {qed_stats['q1']:.6f}")
    print(f"QED Q3     : {qed_stats['q3']:.6f}")
    print("\nQED bin distribution:")
    for k, v in bin_counts.items():
        ratio = v / len(df_unique) if len(df_unique) > 0 else 0
        print(f"  {k}: {int(v)} ({ratio:.2%})")

    print("\n输出文件：")
    print(os.path.join(args.outdir, "qed_unique_summary.json"))
    print(os.path.join(args.outdir, "unique_smiles_qed.csv"))
    print(os.path.join(args.outdir, "qed_hist_with_curve_unique.png"))
    print(os.path.join(args.outdir, "qed_summary.txt"))


if __name__ == "__main__":
    main()