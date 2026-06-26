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


# =========================
# 基础工具
# =========================
def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


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


def make_hist(series, out_png, title, xlabel, bins=40):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return
    plt.figure(figsize=(7, 5), dpi=200)
    plt.hist(s, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


def make_bar(labels, values, out_png, title, xlabel, ylabel, rotate_x=False):
    if len(labels) == 0:
        return
    plt.figure(figsize=(8, 5), dpi=200)
    plt.bar(labels, values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if rotate_x:
        plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


def make_top_barh(df, label_col, value_col, out_png, title, xlabel, ylabel, topn=20, ascending=False):
    if df.empty or label_col not in df.columns or value_col not in df.columns:
        return

    work = df[[label_col, value_col]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=[value_col])
    if work.empty:
        return

    work = work.sort_values(value_col, ascending=ascending).head(topn)

    plt.figure(figsize=(10, 6), dpi=200)
    plt.barh(work[label_col].astype(str), work[value_col])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


def calc_threshold_stats(series, thresholds, prefix="ge"):
    s = pd.to_numeric(series, errors="coerce").dropna()
    result = {
        "n_targets_total": int(len(s))
    }
    for thr in thresholds:
        count = int((s >= thr).sum())
        ratio = float(count / len(s)) if len(s) > 0 else None
        key_base = str(thr).replace(".", "_")
        result[f"n_targets_{prefix}_{key_base}"] = count
        result[f"ratio_targets_{prefix}_{key_base}"] = ratio
    return result


def pick_extreme(df, col, mode="max"):
    work = df[["target", col]].copy()
    work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=[col])

    if work.empty:
        return None

    if mode == "max":
        row = work.sort_values(col, ascending=False).iloc[0]
    else:
        row = work.sort_values(col, ascending=True).iloc[0]

    return {
        "target": str(row["target"]),
        "value": float(row[col])
    }


def make_ratio_bin_distribution(series, labels, bins):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return {}
    cats = pd.cut(s, bins=bins, labels=labels, include_lowest=True, right=True)
    vc = cats.value_counts().reindex(labels, fill_value=0)
    return {str(k): int(v) for k, v in vc.items()}


# =========================
# 主流程
# =========================
def main():
    parser = argparse.ArgumentParser(description="对分子性质统计结果做二次汇总分析")
    parser.add_argument(
        "--input_dir",
        type=str,
        default=".",
        help="上一轮输出目录，里面包含 all_valid_targets_compound_properties.csv"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="分子性质统计文件夹_汇总统计",
        help="新的汇总统计输出目录"
    )
    args = parser.parse_args()

    safe_mkdir(args.outdir)

    compounds_csv = os.path.join(args.input_dir, "all_valid_targets_compound_properties.csv")
    if not os.path.isfile(compounds_csv):
        print(f"[ERR] 找不到输入文件: {compounds_csv}")
        sys.exit(1)

    df = pd.read_csv(compounds_csv)

    required_cols = ["target", "smiles"]
    for c in required_cols:
        if c not in df.columns:
            print(f"[ERR] 输入文件缺少必要列: {c}")
            sys.exit(2)

    numeric_cols = [
        "mol_wt", "logp", "hbd", "hba", "tpsa", "rotatable_bonds",
        "ring_count", "aromatic_ring_count", "heavy_atom_count",
        "fraction_csp3", "molar_refractivity", "qed", "formal_charge",
        "ro5_violations", "lipinski_pass", "veber_pass", "lead_like_pass"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # =========================
    # 1. 全局统计
    # =========================
    global_summary = {
        "n_total_compounds": int(len(df)),
        "n_unique_smiles": int(df["smiles"].nunique()),
        "n_targets": int(df["target"].nunique()),
    }

    stats_cols = [
        "mol_wt", "logp", "tpsa", "qed", "rotatable_bonds",
        "hbd", "hba", "ring_count", "aromatic_ring_count",
        "heavy_atom_count", "fraction_csp3", "molar_refractivity",
        "formal_charge", "ro5_violations"
    ]
    for c in stats_cols:
        if c in df.columns:
            global_summary[c] = series_stats(df[c])

    for c in ["lipinski_pass", "veber_pass", "lead_like_pass"]:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            global_summary[f"{c}_count"] = int(s.sum()) if len(s) > 0 else 0
            global_summary[f"{c}_ratio"] = float(s.mean()) if len(s) > 0 else None

    # 全局不符合五原则比例
    if "lipinski_pass" in df.columns:
        s = pd.to_numeric(df["lipinski_pass"], errors="coerce").dropna()
        if len(s) > 0:
            global_summary["non_lipinski_count"] = int((1 - s).sum())
            global_summary["non_lipinski_ratio"] = float((1 - s).mean())

    # 全局 MW 分箱
    if "mol_wt" in df.columns:
        mw_bins = [0, 200, 300, 400, 500, 600, np.inf]
        mw_labels = ["<=200", "200-300", "300-400", "400-500", "500-600", ">600"]
        df["mw_bin"] = pd.cut(df["mol_wt"], bins=mw_bins, labels=mw_labels, include_lowest=True, right=True)
        mw_counts = df["mw_bin"].value_counts().reindex(mw_labels, fill_value=0)
        global_summary["mw_bin_distribution"] = {k: int(v) for k, v in mw_counts.items()}

    # 全局 Ro5 violation 分布
    if "ro5_violations" in df.columns:
        ro5_counts = df["ro5_violations"].value_counts().sort_index()
        global_summary["ro5_violations_distribution"] = {str(k): int(v) for k, v in ro5_counts.items()}

    save_json(global_summary, os.path.join(args.outdir, "global_summary.json"))

    # =========================
    # 2. 每个靶点统计
    # =========================
    grouped = df.groupby("target", dropna=False)

    target_summary = grouped.agg(
        n_compounds=("smiles", "size"),
        n_unique_smiles=("smiles", "nunique"),

        mol_wt_min=("mol_wt", "min"),
        mol_wt_max=("mol_wt", "max"),
        mol_wt_mean=("mol_wt", "mean"),
        mol_wt_median=("mol_wt", "median"),
        mol_wt_std=("mol_wt", "std"),

        logp_min=("logp", "min"),
        logp_max=("logp", "max"),
        logp_mean=("logp", "mean"),
        logp_median=("logp", "median"),

        tpsa_min=("tpsa", "min"),
        tpsa_max=("tpsa", "max"),
        tpsa_mean=("tpsa", "mean"),
        tpsa_median=("tpsa", "median"),

        qed_min=("qed", "min"),
        qed_max=("qed", "max"),
        qed_mean=("qed", "mean"),
        qed_median=("qed", "median"),

        rotatable_bonds_mean=("rotatable_bonds", "mean"),
        ro5_violations_mean=("ro5_violations", "mean"),

        lipinski_pass_count=("lipinski_pass", "sum"),
        lipinski_pass_ratio=("lipinski_pass", "mean"),

        veber_pass_count=("veber_pass", "sum"),
        veber_pass_ratio=("veber_pass", "mean"),

        lead_like_pass_count=("lead_like_pass", "sum"),
        lead_like_pass_ratio=("lead_like_pass", "mean"),
    ).reset_index()

    target_summary["mol_wt_std"] = target_summary["mol_wt_std"].fillna(0.0)

    # 新增：不符合五原则统计
    if "lipinski_pass_ratio" in target_summary.columns:
        target_summary["non_lipinski_count"] = target_summary["n_compounds"] - target_summary["lipinski_pass_count"]
        target_summary["non_lipinski_ratio"] = 1.0 - target_summary["lipinski_pass_ratio"]

    target_summary = target_summary.sort_values(["n_compounds", "target"], ascending=[False, True]).reset_index(drop=True)

    target_summary_csv = os.path.join(args.outdir, "target_level_summary.csv")
    target_summary.to_csv(target_summary_csv, index=False, encoding="utf-8-sig")

    # =========================
    # 3. 每个靶点的 MW 分箱分布
    # =========================
    target_mw_bin_rows = []
    if "mol_wt" in df.columns:
        mw_bins = [0, 200, 300, 400, 500, 600, np.inf]
        mw_labels = ["<=200", "200-300", "300-400", "400-500", "500-600", ">600"]
        tmp = df[["target", "mol_wt"]].copy()
        tmp["mw_bin"] = pd.cut(tmp["mol_wt"], bins=mw_bins, labels=mw_labels, include_lowest=True, right=True)

        for target, sub in tmp.groupby("target"):
            total = len(sub)
            vc = sub["mw_bin"].value_counts().reindex(mw_labels, fill_value=0)
            row = {
                "target": target,
                "n_compounds": int(total)
            }
            for k, v in vc.items():
                row[f"mw_count_{k}"] = int(v)
                row[f"mw_ratio_{k}"] = float(v / total) if total > 0 else None
            target_mw_bin_rows.append(row)

    target_mw_bin_df = pd.DataFrame(target_mw_bin_rows)
    target_mw_bin_df.to_csv(
        os.path.join(args.outdir, "target_level_mw_bin_distribution.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # =========================
    # 4. 每个靶点 Ro5 violation 分布
    # =========================
    ro5_dist_rows = []
    if "ro5_violations" in df.columns:
        tmp = df[["target", "ro5_violations"]].copy()
        for target, sub in tmp.groupby("target"):
            total = len(sub)
            vc = sub["ro5_violations"].value_counts().sort_index()
            row = {
                "target": target,
                "n_compounds": int(total)
            }
            for k, v in vc.items():
                if pd.isna(k):
                    label = "NA"
                else:
                    try:
                        label = str(int(k))
                    except Exception:
                        label = str(k)
                row[f"ro5_count_{label}"] = int(v)
                row[f"ro5_ratio_{label}"] = float(v / total) if total > 0 else None
            ro5_dist_rows.append(row)

    ro5_dist_df = pd.DataFrame(ro5_dist_rows).fillna(0)
    ro5_dist_df.to_csv(
        os.path.join(args.outdir, "target_level_ro5_distribution.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # =========================
    # 5. 阈值统计
    # =========================
    thresholds = [0.6, 0.7, 0.8]
    low_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

    lipinski_threshold_stats = calc_threshold_stats(target_summary["lipinski_pass_ratio"], thresholds, prefix="ge")
    veber_threshold_stats = calc_threshold_stats(target_summary["veber_pass_ratio"], thresholds, prefix="ge")
    lead_like_threshold_stats = calc_threshold_stats(target_summary["lead_like_pass_ratio"], thresholds, prefix="ge")

    save_json(lipinski_threshold_stats, os.path.join(args.outdir, "lipinski_threshold_stats.json"))
    save_json(veber_threshold_stats, os.path.join(args.outdir, "veber_threshold_stats.json"))
    save_json(lead_like_threshold_stats, os.path.join(args.outdir, "lead_like_threshold_stats.json"))

    # 新增：不符合五原则的阈值统计
    non_lipinski_threshold_stats = calc_threshold_stats(
        target_summary["non_lipinski_ratio"],
        low_thresholds,
        prefix="ge"
    )
    save_json(non_lipinski_threshold_stats, os.path.join(args.outdir, "non_lipinski_threshold_stats.json"))

    # 额外：区间分布
    non_lipinski_bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
    non_lipinski_labels = ["0-10%", "10-20%", "20-30%", "30-40%", "40-50%", ">50%"]
    non_lipinski_bin_distribution = make_ratio_bin_distribution(
        target_summary["non_lipinski_ratio"],
        labels=non_lipinski_labels,
        bins=non_lipinski_bins
    )
    save_json(non_lipinski_bin_distribution, os.path.join(args.outdir, "non_lipinski_bin_distribution.json"))

    # 每个靶点属于哪个不符合比例区间
    target_summary["non_lipinski_ratio_bin"] = pd.cut(
        target_summary["non_lipinski_ratio"],
        bins=non_lipinski_bins,
        labels=non_lipinski_labels,
        include_lowest=True,
        right=True
    )
    target_summary.to_csv(target_summary_csv, index=False, encoding="utf-8-sig")

    # =========================
    # 6. 极值靶点
    # =========================
    extreme_targets = {
        "mol_wt_mean_max": pick_extreme(target_summary, "mol_wt_mean", mode="max"),
        "mol_wt_mean_min": pick_extreme(target_summary, "mol_wt_mean", mode="min"),

        "mol_wt_max_max": pick_extreme(target_summary, "mol_wt_max", mode="max"),
        "mol_wt_min_min": pick_extreme(target_summary, "mol_wt_min", mode="min"),

        "lipinski_pass_ratio_max": pick_extreme(target_summary, "lipinski_pass_ratio", mode="max"),
        "lipinski_pass_ratio_min": pick_extreme(target_summary, "lipinski_pass_ratio", mode="min"),

        "non_lipinski_ratio_max": pick_extreme(target_summary, "non_lipinski_ratio", mode="max"),
        "non_lipinski_ratio_min": pick_extreme(target_summary, "non_lipinski_ratio", mode="min"),

        "veber_pass_ratio_max": pick_extreme(target_summary, "veber_pass_ratio", mode="max"),
        "veber_pass_ratio_min": pick_extreme(target_summary, "veber_pass_ratio", mode="min"),

        "lead_like_pass_ratio_max": pick_extreme(target_summary, "lead_like_pass_ratio", mode="max"),
        "lead_like_pass_ratio_min": pick_extreme(target_summary, "lead_like_pass_ratio", mode="min"),

        "qed_mean_max": pick_extreme(target_summary, "qed_mean", mode="max"),
        "qed_mean_min": pick_extreme(target_summary, "qed_mean", mode="min"),

        "logp_mean_max": pick_extreme(target_summary, "logp_mean", mode="max"),
        "logp_mean_min": pick_extreme(target_summary, "logp_mean", mode="min"),

        "tpsa_mean_max": pick_extreme(target_summary, "tpsa_mean", mode="max"),
        "tpsa_mean_min": pick_extreme(target_summary, "tpsa_mean", mode="min"),

        "n_compounds_max": pick_extreme(target_summary, "n_compounds", mode="max"),
        "n_compounds_min": pick_extreme(target_summary, "n_compounds", mode="min"),
    }
    save_json(extreme_targets, os.path.join(args.outdir, "extreme_targets_summary.json"))

    # =========================
    # 7. Top表
    # =========================
    def save_top_csv(df_in, col, filename, ascending=False, topn=20):
        work = df_in[["target", "n_compounds", col]].copy()
        work[col] = pd.to_numeric(work[col], errors="coerce")
        work = work.dropna(subset=[col]).sort_values(col, ascending=ascending).head(topn)
        work.to_csv(os.path.join(args.outdir, filename), index=False, encoding="utf-8-sig")

    save_top_csv(target_summary, "mol_wt_mean", "top20_mol_wt_mean_high.csv", ascending=False)
    save_top_csv(target_summary, "mol_wt_max", "top20_mol_wt_max_high.csv", ascending=False)
    save_top_csv(target_summary, "lipinski_pass_ratio", "top20_lipinski_pass_ratio_high.csv", ascending=False)
    save_top_csv(target_summary, "lipinski_pass_ratio", "top20_lipinski_pass_ratio_low.csv", ascending=True)
    save_top_csv(target_summary, "non_lipinski_ratio", "top20_non_lipinski_ratio_high.csv", ascending=False)
    save_top_csv(target_summary, "veber_pass_ratio", "top20_veber_pass_ratio_high.csv", ascending=False)
    save_top_csv(target_summary, "lead_like_pass_ratio", "top20_lead_like_pass_ratio_high.csv", ascending=False)
    save_top_csv(target_summary, "qed_mean", "top20_qed_mean_high.csv", ascending=False)

    # =========================
    # 8. 画图
    # =========================
    if "mol_wt" in df.columns:
        make_hist(
            df["mol_wt"],
            os.path.join(args.outdir, "hist_global_mol_wt.png"),
            "Global Molecular Weight Distribution",
            "Molecular Weight (Da)",
            bins=50
        )

    if "lipinski_pass_ratio" in target_summary.columns:
        make_hist(
            target_summary["lipinski_pass_ratio"],
            os.path.join(args.outdir, "hist_target_lipinski_pass_ratio.png"),
            "Target-level Lipinski Pass Ratio Distribution",
            "Lipinski Pass Ratio",
            bins=30
        )

    if "non_lipinski_ratio" in target_summary.columns:
        make_hist(
            target_summary["non_lipinski_ratio"],
            os.path.join(args.outdir, "hist_target_non_lipinski_ratio.png"),
            "Target-level Non-Lipinski Ratio Distribution",
            "Non-Lipinski Ratio",
            bins=30
        )

    if "veber_pass_ratio" in target_summary.columns:
        make_hist(
            target_summary["veber_pass_ratio"],
            os.path.join(args.outdir, "hist_target_veber_pass_ratio.png"),
            "Target-level Veber Pass Ratio Distribution",
            "Veber Pass Ratio",
            bins=30
        )

    if "lead_like_pass_ratio" in target_summary.columns:
        make_hist(
            target_summary["lead_like_pass_ratio"],
            os.path.join(args.outdir, "hist_target_lead_like_pass_ratio.png"),
            "Target-level Lead-like Pass Ratio Distribution",
            "Lead-like Pass Ratio",
            bins=30
        )

    make_top_barh(
        target_summary,
        label_col="target",
        value_col="mol_wt_mean",
        out_png=os.path.join(args.outdir, "top20_mol_wt_mean_high.png"),
        title="Top 20 Targets by Mean Molecular Weight",
        xlabel="Mean Molecular Weight",
        ylabel="Target",
        topn=20,
        ascending=False
    )

    make_top_barh(
        target_summary,
        label_col="target",
        value_col="lipinski_pass_ratio",
        out_png=os.path.join(args.outdir, "top20_lipinski_pass_ratio_high.png"),
        title="Top 20 Targets by Lipinski Pass Ratio",
        xlabel="Lipinski Pass Ratio",
        ylabel="Target",
        topn=20,
        ascending=False
    )

    make_top_barh(
        target_summary,
        label_col="target",
        value_col="lipinski_pass_ratio",
        out_png=os.path.join(args.outdir, "top20_lipinski_pass_ratio_low.png"),
        title="Bottom 20 Targets by Lipinski Pass Ratio",
        xlabel="Lipinski Pass Ratio",
        ylabel="Target",
        topn=20,
        ascending=True
    )

    make_top_barh(
        target_summary,
        label_col="target",
        value_col="non_lipinski_ratio",
        out_png=os.path.join(args.outdir, "top20_non_lipinski_ratio_high.png"),
        title="Top 20 Targets by Non-Lipinski Ratio",
        xlabel="Non-Lipinski Ratio",
        ylabel="Target",
        topn=20,
        ascending=False
    )

    # 阈值靶点数柱状图
    lipinski_thr_labels = [">=0.6", ">=0.7", ">=0.8"]
    lipinski_thr_values = [
        lipinski_threshold_stats["n_targets_ge_0_6"],
        lipinski_threshold_stats["n_targets_ge_0_7"],
        lipinski_threshold_stats["n_targets_ge_0_8"],
    ]
    make_bar(
        lipinski_thr_labels,
        lipinski_thr_values,
        os.path.join(args.outdir, "bar_lipinski_threshold_counts.png"),
        "Number of Targets Passing Lipinski Ratio Thresholds",
        "Threshold",
        "Number of Targets"
    )

    non_lipinski_thr_labels = [">=10%", ">=20%", ">=30%", ">=40%", ">=50%"]
    non_lipinski_thr_values = [
        non_lipinski_threshold_stats["n_targets_ge_0_1"],
        non_lipinski_threshold_stats["n_targets_ge_0_2"],
        non_lipinski_threshold_stats["n_targets_ge_0_3"],
        non_lipinski_threshold_stats["n_targets_ge_0_4"],
        non_lipinski_threshold_stats["n_targets_ge_0_5"],
    ]
    make_bar(
        non_lipinski_thr_labels,
        non_lipinski_thr_values,
        os.path.join(args.outdir, "bar_non_lipinski_threshold_counts.png"),
        "Number of Targets Above Non-Lipinski Ratio Thresholds",
        "Threshold",
        "Number of Targets"
    )

    if len(non_lipinski_bin_distribution) > 0:
        make_bar(
            list(non_lipinski_bin_distribution.keys()),
            list(non_lipinski_bin_distribution.values()),
            os.path.join(args.outdir, "bar_non_lipinski_bin_distribution.png"),
            "Distribution of Target-level Non-Lipinski Ratios",
            "Non-Lipinski Ratio Bin",
            "Number of Targets",
            rotate_x=False
        )

    # =========================
    # 9. 文本摘要
    # =========================
    lines = []
    lines.append("分子性质汇总统计摘要")
    lines.append("=" * 60)
    lines.append(f"总化合物数: {global_summary['n_total_compounds']}")
    lines.append(f"全局唯一 SMILES 数: {global_summary['n_unique_smiles']}")
    lines.append(f"总靶点数: {global_summary['n_targets']}")

    if "mol_wt" in global_summary:
        lines.append("")
        lines.append("全局分子量统计")
        lines.append(f"最小值: {global_summary['mol_wt']['min']:.6f}")
        lines.append(f"最大值: {global_summary['mol_wt']['max']:.6f}")
        lines.append(f"均值: {global_summary['mol_wt']['mean']:.6f}")
        lines.append(f"中位数: {global_summary['mol_wt']['median']:.6f}")
        lines.append(f"标准差: {global_summary['mol_wt']['std']:.6f}")

    lines.append("")
    lines.append("全局规则通过比例")
    lines.append(f"Lipinski 五原则通过比例: {global_summary.get('lipinski_pass_ratio', None)}")
    lines.append(f"不符合五原则比例: {global_summary.get('non_lipinski_ratio', None)}")
    lines.append(f"Veber 通过比例: {global_summary.get('veber_pass_ratio', None)}")
    lines.append(f"Lead-like 通过比例: {global_summary.get('lead_like_pass_ratio', None)}")

    lines.append("")
    lines.append("达到不同五原则比例阈值的靶点数")
    lines.append(f"lipinski_pass_ratio >= 0.6 的靶点数: {lipinski_threshold_stats['n_targets_ge_0_6']} / {lipinski_threshold_stats['n_targets_total']}")
    lines.append(f"lipinski_pass_ratio >= 0.7 的靶点数: {lipinski_threshold_stats['n_targets_ge_0_7']} / {lipinski_threshold_stats['n_targets_total']}")
    lines.append(f"lipinski_pass_ratio >= 0.8 的靶点数: {lipinski_threshold_stats['n_targets_ge_0_8']} / {lipinski_threshold_stats['n_targets_total']}")

    lines.append("")
    lines.append("达到不同不符合五原则比例阈值的靶点数")
    lines.append(f"non_lipinski_ratio >= 0.1 的靶点数: {non_lipinski_threshold_stats['n_targets_ge_0_1']} / {non_lipinski_threshold_stats['n_targets_total']}")
    lines.append(f"non_lipinski_ratio >= 0.2 的靶点数: {non_lipinski_threshold_stats['n_targets_ge_0_2']} / {non_lipinski_threshold_stats['n_targets_total']}")
    lines.append(f"non_lipinski_ratio >= 0.3 的靶点数: {non_lipinski_threshold_stats['n_targets_ge_0_3']} / {non_lipinski_threshold_stats['n_targets_total']}")
    lines.append(f"non_lipinski_ratio >= 0.4 的靶点数: {non_lipinski_threshold_stats['n_targets_ge_0_4']} / {non_lipinski_threshold_stats['n_targets_total']}")
    lines.append(f"non_lipinski_ratio >= 0.5 的靶点数: {non_lipinski_threshold_stats['n_targets_ge_0_5']} / {non_lipinski_threshold_stats['n_targets_total']}")

    lines.append("")
    lines.append("不符合五原则比例区间分布")
    for k, v in non_lipinski_bin_distribution.items():
        lines.append(f"{k}: {v}")

    lines.append("")
    lines.append("关键极值靶点")
    for key in [
        "mol_wt_mean_max",
        "mol_wt_max_max",
        "mol_wt_min_min",
        "lipinski_pass_ratio_max",
        "lipinski_pass_ratio_min",
        "non_lipinski_ratio_max",
        "non_lipinski_ratio_min",
        "veber_pass_ratio_max",
        "lead_like_pass_ratio_max",
        "qed_mean_max",
        "logp_mean_max",
        "tpsa_mean_max",
        "n_compounds_max",
    ]:
        item = extreme_targets.get(key)
        if item is not None:
            lines.append(f"{key}: {item['target']} -> {item['value']:.6f}")

    with open(os.path.join(args.outdir, "summary_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # =========================
    # 10. 运行信息
    # =========================
    run_info = {
        "command": " ".join(sys.argv),
        "input_dir": os.path.abspath(args.input_dir),
        "input_csv": os.path.abspath(compounds_csv),
        "outdir": os.path.abspath(args.outdir),
        "n_total_compounds": int(len(df)),
        "n_targets": int(df["target"].nunique()),
    }
    save_json(run_info, os.path.join(args.outdir, "run_info.json"))

    print("\n===== 完成 =====")
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.outdir}")
    print(f"靶点主表: {os.path.join(args.outdir, 'target_level_summary.csv')}")
    print(f"Lipinski 阈值统计: {os.path.join(args.outdir, 'lipinski_threshold_stats.json')}")
    print(f"Non-Lipinski 阈值统计: {os.path.join(args.outdir, 'non_lipinski_threshold_stats.json')}")
    print(f"Non-Lipinski 区间分布: {os.path.join(args.outdir, 'non_lipinski_bin_distribution.json')}")
    print(f"文本摘要: {os.path.join(args.outdir, 'summary_report.txt')}")


if __name__ == "__main__":
    main()