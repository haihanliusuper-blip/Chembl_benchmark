# -*- coding: utf-8 -*-
import os
import sys
import csv
import json
import argparse
import traceback
from collections import Counter

import pandas as pd
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np


# ========= 分子指纹 =========
def smiles_to_fingerprint(smiles, radius=2, n_bits=2048):
    """
    使用 RDKit 提取分子指纹（Morgan指纹）
    :param smiles: SMILES 字符串
    :param radius: Morgan 指纹的半径，默认为2
    :param n_bits: 分子指纹的位数，默认为2048
    :return: 指纹（ndarray 格式）
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # 使用 Morgan 指纹生成位向量
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    # 转换为 numpy 数组
    fp_array = np.array(fp, dtype=int)
    return fp_array


# ========= 处理单个 CSV =========
def process_ic50_csv(csv_path, smiles_col='compound_smiles', value_col='value_num', unit_col='value_units'):
    df = pd.read_csv(csv_path)
    missing = [c for c in [smiles_col, value_col, unit_col] if c not in df.columns]
    if missing:
        raise ValueError(f"列缺失: {missing} in {csv_path}")

    units = df[unit_col].astype(str).str.strip().fillna("NA")
    uniq_units = sorted(units.unique().tolist())
    units_ok = len(uniq_units) == 1

    vals = pd.to_numeric(df[value_col], errors='coerce')
    mean_thr = float(vals.mean())

    labels = (vals > mean_thr).astype(int)

    fingerprint_list, kept_smiles, kept_labels = [], [], []
    bad_rows = 0

    for smi, y in zip(df[smiles_col].astype(str), labels):
        # 先快速验证 SMILES 是否能被 RDKit 解析
        mol = Chem.MolFromSmiles(smi)
        if mol is None or mol.GetNumAtoms() == 0:
            bad_rows += 1
            continue

        # 使用 RDKit 提取分子指纹
        fp = smiles_to_fingerprint(smi)
        if fp is None:
            bad_rows += 1
            continue

        fingerprint_list.append(fp)
        kept_smiles.append(smi)
        kept_labels.append(int(y))

    payload = {
        'fingerprints': fingerprint_list,
        'x_smiles': kept_smiles,
        'labels': kept_labels,
        'threshold_mean_value_num': mean_thr,
        'value_units_unique': uniq_units,
        'units_consistent': units_ok,
        'source_csv': os.path.abspath(csv_path),
        'skipped_bad_smiles': bad_rows,
        'total_rows': int(len(df))
    }

    return payload, Counter(kept_labels), units_ok


# ========= 主流程 =========
def main():
    parser = argparse.ArgumentParser(description="扫描子目录的 IC50.CSV 生成 IC50_mean.pt 并输出统计 CSV")
    parser.add_argument("--root", type=str, default="ChEMBL_Targets_MIN", help="根目录")
    parser.add_argument("--csv_name", type=str, default="IC50.CSV", help="要查找的 CSV 文件名")
    parser.add_argument("--smiles_col", type=str, default="compound_smiles", help="SMILES 列名")
    parser.add_argument("--value_col", type=str, default="value_num", help="IC50 数值列名")
    parser.add_argument("--unit_col", type=str, default="value_units", help="单位列名")
    parser.add_argument("--summary_csv", type=str, default="IC50_mean_summary.csv", help="统计输出 CSV 文件名")
    args = parser.parse_args()

    root = args.root
    if not os.path.isdir(root):
        print(f"[ERR] 根目录不存在: {root}")
        sys.exit(1)

    per_folder_csv_cnt = {}
    per_folder_label_stats = {}
    per_folder_units_inconsistent = {}

    total_label_counter = Counter()
    total_csv_counter = 0
    failed_files = []

    for folder_name in sorted(os.listdir(root)):
        sub = os.path.join(root, folder_name)
        if not os.path.isdir(sub):
            continue

        csvs = [os.path.join(sub, fn) for fn in os.listdir(sub) if fn.lower() == args.csv_name.lower()]
        per_folder_csv_cnt[folder_name] = len(csvs)

        folder_label_counter = Counter()
        folder_units_ok_all = True

        for csv_path in csvs:
            try:
                payload, label_counter, units_ok = process_ic50_csv(
                    csv_path,
                    smiles_col=args.smiles_col,
                    value_col=args.value_col,
                    unit_col=args.unit_col
                )

                # 保存为 .npz 文件
                out_npz = os.path.join(os.path.dirname(csv_path), "IC50_mean.npz")
                np.savez(out_npz, fingerprints=payload['fingerprints'], labels=payload['labels'])

                folder_label_counter.update(label_counter)
                total_label_counter.update(label_counter)
                total_csv_counter += 1
                if not units_ok:
                    folder_units_ok_all = False

                print(
                    f"[OK] 保存: {out_npz} 样本数={sum(label_counter.values())} 阈值={payload['threshold_mean_value_num']:.6g}")
            except Exception as e:
                failed_files.append(csv_path)
                print(f"[FAIL] {csv_path}\n{e}")
                traceback.print_exc()

        per_folder_label_stats[folder_name] = dict(folder_label_counter)
        per_folder_units_inconsistent[folder_name] = not folder_units_ok_all

    print("\n===== 子文件夹统计 =====")
    for folder in sorted(per_folder_csv_cnt.keys()):
        n_csv = per_folder_csv_cnt[folder]
        stats = per_folder_label_stats.get(folder, {})
        warn = per_folder_units_inconsistent.get(folder, False)
        warn_txt = " | 单位存在不一致" if warn else ""
        print(f"{folder}: CSV数={n_csv} | label0={stats.get(0, 0)} | label1={stats.get(1, 0)}{warn_txt}")

    print("\n===== 全局统计 =====")
    print(
        f"总CSV数={total_csv_counter} | 总label0={total_label_counter.get(0, 0)} | 总label1={total_label_counter.get(1, 0)}")

    if failed_files:
        print("\n===== 处理失败的文件 =====")
        for p in failed_files:
            print(p)

    # 保存统计到 CSV
    csv_path = args.summary_csv
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["folder", "csv_count", "label0", "label1", "units_inconsistent"])
        for folder in sorted(per_folder_csv_cnt.keys()):
            writer.writerow([
                folder,
                per_folder_csv_cnt[folder],
                per_folder_label_stats.get(folder, {}).get(0, 0),
                per_folder_label_stats.get(folder, {}).get(1, 0),
                per_folder_units_inconsistent.get(folder, False)
            ])
        writer.writerow([
            "__TOTAL__",
            total_csv_counter,
            total_label_counter.get(0, 0),
            total_label_counter.get(1, 0),
            ""
        ])

    # 同时保存一份失败清单
    fail_json = os.path.splitext(csv_path)[0] + "_failed.json"
    with open(fail_json, "w", encoding="utf-8") as f:
        json.dump({"failed_files": failed_files}, f, ensure_ascii=False, indent=2)

    print(f"\n[保存] 统计 CSV: {csv_path}")
    print(f"[保存] 失败清单: {fail_json}")


if __name__ == "__main__":
    main()
