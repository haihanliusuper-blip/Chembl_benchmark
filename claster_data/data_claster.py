#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
一键跑两种划分方案的小助手脚本：

1. cluster_holdout 方案
   输出：IC50.holdout.csv

2. cluster_stratified 方案
   输出：IC50.stratified.csv

用法示例：
  python run_cluster_splits.py --root_dir "ChEMBL_Targets_MIN"

默认假设：
  - cluster_split_ic50.py 和本脚本在同一目录
  - 子目录里的文件名是 IC50.csv
  - SMILES 列名是 Smiles
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd_list):
    """包装一下 subprocess 调用"""
    print("\n>>> 运行命令：")
    print(" ", " ".join(cmd_list))
    try:
        subprocess.check_call(cmd_list)
    except subprocess.CalledProcessError as e:
        print("  [ERROR] 命令执行失败，返回码:", e.returncode)
        sys.exit(e.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="批量运行 cluster_split_ic50.py 的两种划分方案"
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="ChEMBL_Targets_MIN 根目录路径"
    )
    parser.add_argument(
        "--csv_name",
        type=str,
        default="IC50.csv",
        help="子目录中要处理的 CSV 文件名"
    )
    parser.add_argument(
        "--smiles_col",
        type=str,
        default="compound_smiles",
        help="SMILES 列名"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Tanimoto 相似度阈值"
    )
    parser.add_argument(
        "--min_group_size",
        type=int,
        default=10,
        help="有效簇的最小大小"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="cluster_stratified 下 train 占比"
    )
    parser.add_argument(
        "--cluster_test_frac",
        type=float,
        default=0.3,
        help="cluster_holdout 下测试簇的比例"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="随机种子"
    )
    parser.add_argument(
        "--python_exec",
        type=str,
        default=sys.executable,
        help="Python 解释器路径，默认使用当前解释器"
    )
    args = parser.parse_args()

    script_path = Path(__file__).parent / "cluster_split_ic50.py"
    if not script_path.exists():
        print("[ERROR] 找不到 cluster_split_ic50.py，确认它和本脚本在同一目录下。")
        sys.exit(1)

    # 公共参数
    base_cmd = [
        args.python_exec,
        str(script_path),
        "--root_dir",
        args.root_dir,
        "--csv_name",
        args.csv_name,
        "--smiles_col",
        args.smiles_col,
        "--threshold",
        str(args.threshold),
        "--min_group_size",
        str(args.min_group_size),
        "--train_ratio",
        str(args.train_ratio),
        "--cluster_test_frac",
        str(args.cluster_test_frac),
        "--seed",
        str(args.seed),
    ]

    # 1. cluster_holdout
    print("\n==============================")
    print("  开始运行方案一：cluster_holdout")
    print("==============================")
    cmd_holdout = base_cmd + [
        "--mode",
        "cluster_holdout",
        "--suffix",
        ".holdout",
    ]
    run_cmd(cmd_holdout)

    # 2. cluster_stratified
    print("\n==============================")
    print("  开始运行方案二：cluster_stratified")
    print("==============================")
    cmd_stratified = base_cmd + [
        "--mode",
        "cluster_stratified",
        "--suffix",
        ".stratified",
    ]
    run_cmd(cmd_stratified)

    print("\n全部完成，去各个靶点文件夹里检查：")
    print("  - IC50.holdout.csv")
    print("  - IC50.stratified.csv")


if __name__ == "__main__":
    main()
