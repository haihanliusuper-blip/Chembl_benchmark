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
from rdkit.Chem import GetAdjacencyMatrix
from torch_geometric.data import Data
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import argparse
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import torch.nn.functional as F
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from rdkit import Chem
from rdkit.Chem import PeriodicTable
from torch.utils.data import DataLoader, random_split

from rdkit.Chem import GetAdjacencyMatrix
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import pandas as pd
import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data
# ========= 分子特征 =========
def one_hot_encoding(value, choices):
    return [1 if choice == value else 0 for choice in choices]

def get_atom_features(atom, use_chirality=True, hydrogens_implicit=True):
    permitted_atoms = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Unknown']
    if not hydrogens_implicit:
        permitted_atoms = ['H'] + permitted_atoms

    atom_type = str(atom.GetSymbol())
    if atom_type not in permitted_atoms:
        atom_type = 'Unknown'

    atom_type_enc = one_hot_encoding(atom_type, permitted_atoms)

    deg = atom.GetDegree()
    n_heavy_neighbors_enc = one_hot_encoding(deg if deg <= 4 else "MoreThanFour",
                                            [0, 1, 2, 3, 4, "MoreThanFour"])

    fc = atom.GetFormalCharge()
    formal_charge_enc = one_hot_encoding(fc if -3 <= fc <= 3 else "Extreme",
                                         [-3, -2, -1, 0, 1, 2, 3, "Extreme"])

    hybridisation_type_enc = one_hot_encoding(str(atom.GetHybridization()),
                                              ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])

    is_in_a_ring_enc = [int(atom.IsInRing())]
    is_aromatic_enc = [int(atom.GetIsAromatic())]

    pt = Chem.GetPeriodicTable()
    vdw_radius_scaled = [float((pt.GetRvdw(atom.GetAtomicNum()) - 1.5) / 0.6)]
    covalent_radius_scaled = [float((pt.GetRcovalent(atom.GetAtomicNum()) - 0.64) / 0.76)]

    feat = atom_type_enc + n_heavy_neighbors_enc + formal_charge_enc + hybridisation_type_enc \
           + is_in_a_ring_enc + is_aromatic_enc + vdw_radius_scaled + covalent_radius_scaled

    if use_chirality:
        feat += one_hot_encoding(str(atom.GetChiralTag()),
                                 ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])

    if hydrogens_implicit:
        nH = atom.GetTotalNumHs()
        feat += one_hot_encoding(nH if nH <= 4 else "MoreThanFour", [0, 1, 2, 3, 4, "MoreThanFour"])

    return np.array(feat, dtype=float)

def get_bond_features(bond, use_stereochemistry=True):
    permitted = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                 Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    bond_type_enc = one_hot_encoding(bond.GetBondType(), permitted)
    bond_is_conj_enc = [int(bond.GetIsConjugated())]
    bond_is_in_ring_enc = [int(bond.IsInRing())]
    feat = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc
    if use_stereochemistry:
        feat += one_hot_encoding(str(bond.GetStereo()),
                                 ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
    return np.array(feat, dtype=float)

def smiles_to_data(smiles, y_val):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    n_nodes = mol.GetNumAtoms()
    if n_nodes == 0:
        return None

    probe = Chem.MolFromSmiles("O=O")
    n_node_features = len(get_atom_features(probe.GetAtomWithIdx(0)))
    X = np.zeros((n_nodes, n_node_features), dtype=float)
    for atom in mol.GetAtoms():
        X[atom.GetIdx(), :] = get_atom_features(atom)
    X = torch.tensor(X, dtype=torch.float)

    adj = GetAdjacencyMatrix(mol)
    rows, cols = np.nonzero(adj)
    edge_index = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)

    if mol.GetNumBonds() == 0:
        n_edge_features = len(get_bond_features(probe.GetBondBetweenAtoms(0, 1)))
        EF = torch.zeros((0, n_edge_features), dtype=torch.float)
    else:
        n_edge_features = len(get_bond_features(mol.GetBondBetweenAtoms(0, 1)))
        ef_np = np.zeros((edge_index.shape[1], n_edge_features), dtype=float)
        for k, (i, j) in enumerate(zip(rows, cols)):
            bond = mol.GetBondBetweenAtoms(int(i), int(j))
            if bond is None:
                ef_np[k] = np.zeros((n_edge_features,), dtype=float)
            else:
                ef_np[k] = get_bond_features(bond)
        EF = torch.tensor(ef_np, dtype=torch.float)

    data = Data(
        x=X,
        edge_index=edge_index,
        edge_attr=EF,
        y=torch.tensor([float(y_val)], dtype=torch.float)
    )
    return data

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

    data_list, kept_smiles, kept_labels = [], [], []
    bad_rows = 0

    for smi, y in zip(df[smiles_col].astype(str), labels):
        # 先快速验证 SMILES 是否能被 RDKit 解析
        mol = Chem.MolFromSmiles(smi)
        if mol is None or mol.GetNumAtoms() == 0:
            bad_rows += 1
            # 可选：打印一行提示，避免刷屏可注释掉
            # print(f"[跳过] 无法解析 SMILES: {smi}")
            continue

        # 保持 smiles_to_data 原样不变；若其内部因特殊结构抛错，这里捕获并跳过
        try:
            d = smiles_to_data(smi, int(y))
        except Exception as e:
            bad_rows += 1
            # 可选：打印一行提示，定位问题 SMILES
            # print(f"[跳过] smiles_to_data 失败: {smi} | 错误: {e}")
            continue

        if d is None:
            bad_rows += 1
            continue

        data_list.append(d)
        kept_smiles.append(smi)
        kept_labels.append(int(y))

    payload = {
        'data_list': data_list,
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

        # 查找 IC50.CSV（大小写不敏感）
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
                out_pt = os.path.join(os.path.dirname(csv_path), "IC50_mean.pt")
                torch.save(payload, out_pt)

                folder_label_counter.update(label_counter)
                total_label_counter.update(label_counter)
                total_csv_counter += 1
                if not units_ok:
                    folder_units_ok_all = False

                print(f"[OK] 保存: {out_pt} 样本数={sum(label_counter.values())} 阈值={payload['threshold_mean_value_num']:.6g}")
            except Exception as e:
                failed_files.append(csv_path)
                print(f"[FAIL] {csv_path}\n{e}")
                traceback.print_exc()

        per_folder_label_stats[folder_name] = dict(folder_label_counter)
        per_folder_units_inconsistent[folder_name] = not folder_units_ok_all

    # 打印统计
    print("\n===== 子文件夹统计 =====")
    for folder in sorted(per_folder_csv_cnt.keys()):
        n_csv = per_folder_csv_cnt[folder]
        stats = per_folder_label_stats.get(folder, {})
        warn = per_folder_units_inconsistent.get(folder, False)
        warn_txt = " | 单位存在不一致" if warn else ""
        print(f"{folder}: CSV数={n_csv} | label0={stats.get(0,0)} | label1={stats.get(1,0)}{warn_txt}")

    print("\n===== 全局统计 =====")
    print(f"总CSV数={total_csv_counter} | 总label0={total_label_counter.get(0,0)} | 总label1={total_label_counter.get(1,0)}")

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
        # 也写入一行全局合计，folder 用 "__TOTAL__"
        writer.writerow([
            "__TOTAL__",
            total_csv_counter,
            total_label_counter.get(0, 0),
            total_label_counter.get(1, 0),
            ""  # 合计不适用单位一致性
        ])

    # 同时保存一份失败清单以便追溯
    fail_json = os.path.splitext(csv_path)[0] + "_failed.json"
    with open(fail_json, "w", encoding="utf-8") as f:
        json.dump({"failed_files": failed_files}, f, ensure_ascii=False, indent=2)

    print(f"\n[保存] 统计 CSV: {csv_path}")
    print(f"[保存] 失败清单: {fail_json}")

if __name__ == "__main__":
    main()
