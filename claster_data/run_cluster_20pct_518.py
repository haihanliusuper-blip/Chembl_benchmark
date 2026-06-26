#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
一键重跑 20% cluster split 全流程，并把本次结果统一存放到：按簇划分_518

流程：
1. 调用 cluster_split_ic50.py 生成新的 IC50.holdout.csv 和 IC50.stratified.csv
   - threshold = 0.6
   - min_group_size = 10
   - train_ratio = 0.8
   - cluster_test_frac = 0.2

2. 备份每个靶点下新生成的 IC50.holdout.csv / IC50.stratified.csv 到：
   按簇划分_518/split_csv_backup/

3. 调用 生成分簇数据集.py，把 IC50.holdout.csv 转成：
   IC50.holdout.train.pt
   IC50.holdout.test.pt

4. 调用 训练GIN_分簇.py 训练 GIN，输出：
   按簇划分_518/clusterpt/report_gin_clusterpt.csv

用法示例：
python run_cluster_20pct_518.py ^
  --root_dir ChEMBL_Targets_MIN ^
  --valid_csv report_gin_valid_auc.csv

如果脚本和其他 py 文件不在同一目录，可以手动指定：
python run_cluster_20pct_518.py ^
  --root_dir ChEMBL_Targets_MIN ^
  --cluster_split_script cluster_split_ic50.py ^
  --csv_to_pt_script 生成分簇数据集.py ^
  --train_script 训练GIN_分簇.py ^
  --valid_csv report_gin_valid_auc.csv
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
from pathlib import Path
from collections import Counter

import pandas as pd


def run_cmd(cmd, log_file=None):
    print("\n>>> 运行命令：")
    print(" ".join(str(x) for x in cmd))

    if log_file is None:
        subprocess.check_call(cmd)
        return

    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    with open(log_file, "w", encoding="utf-8") as f:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        for line in process.stdout:
            print(line, end="")
            f.write(line)

        ret = process.wait()

    if ret != 0:
        raise subprocess.CalledProcessError(ret, cmd)


def ensure_exists(path, name):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"[ERROR] 找不到 {name}: {path}")
    return path


def backup_split_csv(root_dir, out_backup_dir, csv_names=("IC50.holdout.csv", "IC50.stratified.csv")):
    root_dir = Path(root_dir)
    out_backup_dir = Path(out_backup_dir)
    out_backup_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for target_dir in sorted(root_dir.iterdir()):
        if not target_dir.is_dir():
            continue

        target_name = target_dir.name
        backup_target_dir = out_backup_dir / target_name
        copied_any = False

        for csv_name in csv_names:
            src = target_dir / csv_name
            if not src.is_file():
                continue

            backup_target_dir.mkdir(parents=True, exist_ok=True)
            dst = backup_target_dir / csv_name
            shutil.copy2(src, dst)

            copied_any = True
            rows.append({
                "target": target_name,
                "csv_name": csv_name,
                "source": str(src.resolve()),
                "backup": str(dst.resolve()),
            })

        if copied_any:
            print(f"[BACKUP] {target_name}")

    df = pd.DataFrame(rows)
    summary_path = out_backup_dir / "backup_summary.csv"
    df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\n[保存] CSV 备份清单: {summary_path}")
    return summary_path


def check_split_ratio(root_dir, csv_name, out_csv):
    root_dir = Path(root_dir)
    rows = []
    total = Counter()

    for target_dir in sorted(root_dir.iterdir()):
        if not target_dir.is_dir():
            continue

        path = target_dir / csv_name
        if not path.is_file():
            continue

        try:
            df = pd.read_csv(path)
        except Exception as e:
            rows.append({
                "target": target_dir.name,
                "status": "read_failed",
                "error": str(e),
            })
            continue

        if "split" not in df.columns:
            rows.append({
                "target": target_dir.name,
                "status": "missing_split_col",
                "error": "no split column",
            })
            continue

        split_series = df["split"].astype(str).str.strip().str.lower()
        c = Counter(split_series)

        n_train = c.get("train", 0)
        n_test = c.get("test", 0)
        n_total = n_train + n_test

        total.update({
            "train": n_train,
            "test": n_test,
        })

        rows.append({
            "target": target_dir.name,
            "status": "ok",
            "n_total": n_total,
            "n_train": n_train,
            "n_test": n_test,
            "molecule_level_test_frac": n_test / n_total if n_total > 0 else None,
        })

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")

    n_train_all = total.get("train", 0)
    n_test_all = total.get("test", 0)
    n_all = n_train_all + n_test_all

    global_summary = {
        "csv_name": csv_name,
        "global_train": n_train_all,
        "global_test": n_test_all,
        "global_total": n_all,
        "global_molecule_level_test_frac": n_test_all / n_all if n_all > 0 else None,
        "note": "这里统计的是 molecule-level 比例。由于划分单位是 cluster，分子数量比例不一定严格等于 0.2。",
    }

    json_path = out_csv.with_suffix(".global_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(global_summary, f, ensure_ascii=False, indent=2)

    print(f"\n[保存] split 比例检查: {out_csv}")
    print(f"[保存] 全局比例 JSON: {json_path}")
    print("[全局统计]", global_summary)

    return out_csv, json_path


def main():
    parser = argparse.ArgumentParser(
        description="一键重跑 20% cluster split，并统一保存到 按簇划分_518"
    )

    parser.add_argument("--root_dir", type=str, required=True,
                        help="原始 ChEMBL_Targets_MIN 根目录")
    parser.add_argument("--out_dir", type=str, default="按簇划分_518",
                        help="本次统一输出目录")

    parser.add_argument("--csv_name", type=str, default="IC50.csv")
    parser.add_argument("--smiles_col", type=str, default="compound_smiles")
    parser.add_argument("--value_col", type=str, default="value_num")
    parser.add_argument("--unit_col", type=str, default="value_units")
    parser.add_argument("--split_col", type=str, default="split")

    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--min_group_size", type=int, default=10)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--cluster_test_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2025)

    parser.add_argument("--valid_csv", type=str, default="report_gin_valid_auc.csv",
                        help="训练脚本需要的 valid target csv")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim1", type=int, default=128)
    parser.add_argument("--hidden_dim2", type=int, default=256)
    parser.add_argument("--dropout_rate", type=float, default=0.5)
    parser.add_argument("--min_size", type=int, default=20)

    parser.add_argument("--python_exec", type=str, default=sys.executable)

    parser.add_argument("--cluster_split_script", type=str, default="cluster_split_ic50.py",
                        help="真正生成 IC50.holdout.csv 的脚本")
    parser.add_argument("--csv_to_pt_script", type=str, default="生成分簇数据集.py",
                        help="把 IC50.holdout.csv 转成 .pt 的脚本")
    parser.add_argument("--train_script", type=str, default="训练GIN_分簇_2.py",
                        help="训练 GIN 分簇数据的脚本")

    parser.add_argument("--skip_split", action="store_true",
                        help="跳过生成 IC50.holdout.csv / IC50.stratified.csv")
    parser.add_argument("--skip_pt", action="store_true",
                        help="跳过 CSV 转 pt")
    parser.add_argument("--skip_train", action="store_true",
                        help="跳过训练")
    parser.add_argument("--skip_backup", action="store_true",
                        help="跳过备份 split csv")

    args = parser.parse_args()

    root_dir = ensure_exists(args.root_dir, "root_dir")
    valid_csv = ensure_exists(args.valid_csv, "valid_csv")

    out_dir = Path(args.out_dir)
    logs_dir = out_dir / "logs"
    backup_dir = out_dir / "split_csv_backup"
    clusterpt_dir = out_dir / "clusterpt"
    checks_dir = out_dir / "checks"

    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    clusterpt_dir.mkdir(parents=True, exist_ok=True)
    checks_dir.mkdir(parents=True, exist_ok=True)

    cluster_split_script = ensure_exists(args.cluster_split_script, "cluster_split_script")
    csv_to_pt_script = ensure_exists(args.csv_to_pt_script, "csv_to_pt_script")
    train_script = ensure_exists(args.train_script, "train_script")

    config = vars(args).copy()
    config["root_dir_abs"] = str(root_dir.resolve())
    config["out_dir_abs"] = str(out_dir.resolve())
    config["clusterpt_dir_abs"] = str(clusterpt_dir.resolve())
    config_path = out_dir / "run_config.json"

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("\n==============================")
    print("本次运行配置")
    print("==============================")
    print(json.dumps(config, ensure_ascii=False, indent=2))
    print(f"\n[保存] 配置文件: {config_path}")

    # 1. 重新生成 cluster split CSV
    if not args.skip_split:
        base_cmd = [
            args.python_exec,
            str(cluster_split_script),
            "--root_dir", str(root_dir),
            "--csv_name", args.csv_name,
            "--smiles_col", args.smiles_col,
            "--threshold", str(args.threshold),
            "--min_group_size", str(args.min_group_size),
            "--train_ratio", str(args.train_ratio),
            "--cluster_test_frac", str(args.cluster_test_frac),
            "--seed", str(args.seed),
        ]

        print("\n==============================")
        print("步骤 1A：生成 20% cluster_holdout")
        print("==============================")
        cmd_holdout = base_cmd + [
            "--mode", "cluster_holdout",
            "--suffix", ".holdout",
        ]
        run_cmd(cmd_holdout, log_file=logs_dir / "01_cluster_holdout.log")

        print("\n==============================")
        print("步骤 1B：生成 cluster_stratified")
        print("==============================")
        cmd_stratified = base_cmd + [
            "--mode", "cluster_stratified",
            "--suffix", ".stratified",
        ]
        run_cmd(cmd_stratified, log_file=logs_dir / "02_cluster_stratified.log")
    else:
        print("\n[SKIP] 跳过 split CSV 生成")

    # 2. 备份 split CSV
    if not args.skip_backup:
        print("\n==============================")
        print("步骤 2：备份 split CSV")
        print("==============================")
        backup_split_csv(
            root_dir=root_dir,
            out_backup_dir=backup_dir,
            csv_names=("IC50.holdout.csv", "IC50.stratified.csv"),
        )
    else:
        print("\n[SKIP] 跳过 split CSV 备份")

    # 3. 检查 split 比例
    print("\n==============================")
    print("步骤 3：检查 holdout split 分子数量比例")
    print("==============================")
    check_split_ratio(
        root_dir=root_dir,
        csv_name="IC50.holdout.csv",
        out_csv=checks_dir / "check_IC50_holdout_split_ratio.csv",
    )

    # 4. CSV 转 pt
    if not args.skip_pt:
        print("\n==============================")
        print("步骤 4：IC50.holdout.csv 转 pt")
        print("==============================")
        cmd_pt = [
            args.python_exec,
            str(csv_to_pt_script),
            "--root_in", str(root_dir),
            "--root_out", str(clusterpt_dir),
            "--csv_name", "IC50.holdout.csv",
            "--smiles_col", args.smiles_col,
            "--value_col", args.value_col,
            "--unit_col", args.unit_col,
            "--split_col", args.split_col,
            "--summary_csv", "IC50_mean_cluster_summary_20pct_518.csv",
        ]
        run_cmd(cmd_pt, log_file=logs_dir / "03_csv_to_pt_holdout.log")
    else:
        print("\n[SKIP] 跳过 CSV 转 pt")

    # 5. 训练 GIN
    if not args.skip_train:
        print("\n==============================")
        print("步骤 5：训练 GIN cluster holdout")
        print("==============================")
        cmd_train = [
            args.python_exec,
            str(train_script),
            "--root", str(clusterpt_dir),
            "--train_pt", "IC50.holdout.train.pt",
            "--test_pt", "IC50.holdout.test.pt",
            "--valid_csv", str(valid_csv),
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--lr", str(args.lr),
            "--hidden_dim1", str(args.hidden_dim1),
            "--hidden_dim2", str(args.hidden_dim2),
            "--dropout_rate", str(args.dropout_rate),
            "--min_size", str(args.min_size),
        ]
        run_cmd(cmd_train, log_file=logs_dir / "04_train_gin_cluster_holdout.log")
    else:
        print("\n[SKIP] 跳过训练")

    print("\n==============================")
    print("全部完成")
    print("==============================")
    print(f"统一输出目录: {out_dir.resolve()}")
    print(f"日志目录: {logs_dir.resolve()}")
    print(f"CSV 备份目录: {backup_dir.resolve()}")
    print(f"pt 和训练报告目录: {clusterpt_dir.resolve()}")
    print(f"检查文件目录: {checks_dir.resolve()}")
    print("\n重点看这些文件：")
    print(f"  {clusterpt_dir / 'report_gin_clusterpt.csv'}")
    print(f"  {clusterpt_dir / 'IC50_mean_cluster_summary_20pct_518.csv'}")
    print(f"  {checks_dir / 'check_IC50_holdout_split_ratio.csv'}")
    print(f"  {checks_dir / 'check_IC50_holdout_split_ratio.global_summary.json'}")


if __name__ == "__main__":
    main()