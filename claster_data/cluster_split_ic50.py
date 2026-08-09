#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Create target-wise Butina cluster-holdout splits from IC50 CSV files."""

import argparse
import csv
import os
import random

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina


def fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def butina_groups(fps, similarity_threshold):
    """Return one group ID for every fingerprint; same group means same Butina cluster."""
    n = len(fps)
    if n == 1:
        return [0]
    distances = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        distances.extend(1.0 - sim for sim in sims)
    clusters = Butina.ClusterData(
        distances, n, 1.0 - similarity_threshold, isDistData=True, reordering=True
    )
    groups = [-1] * n
    for group_id, members in enumerate(clusters):
        for member in members:
            groups[member] = group_id
    return groups


def choose_test_groups(groups, test_fraction, rng):
    unique_groups = sorted(set(groups))
    if len(unique_groups) < 2:
        return None
    group_sizes = {group: groups.count(group) for group in unique_groups}
    desired = max(1, round(len(groups) * test_fraction))
    order = unique_groups[:]
    rng.shuffle(order)
    selected, count = set(), 0
    for group in order:
        if count < desired:
            selected.add(group)
            count += group_sizes[group]
    if len(selected) == len(unique_groups):
        selected.remove(order[-1])
    return selected


def process_one(csv_path, out_path, args, seed):
    source = pd.read_csv(csv_path)
    required = [args.smiles_col, args.value_col]
    missing = [col for col in required if col not in source.columns]
    if missing:
        raise ValueError(f"missing columns {missing}")

    values = pd.to_numeric(source[args.value_col], errors="coerce")
    threshold = float(values.mean())
    rows, fps = [], []
    for original_index, (smiles, value) in enumerate(zip(source[args.smiles_col], values)):
        if pd.isna(value) or pd.isna(smiles):
            continue
        smiles = str(smiles).strip()
        fp = fingerprint(smiles)
        if fp is None:
            continue
        row = source.iloc[original_index].copy()
        row["source_row"] = original_index
        # Keep the label definition used by process_ic50_to_pt.py.
        row["label"] = int(value > threshold)
        rows.append(row)
        fps.append(fp)

    if len(rows) < args.min_molecules:
        return {"status": "skipped_size", "n_valid": len(rows), "threshold": threshold}
    groups = butina_groups(fps, args.threshold)
    test_groups = choose_test_groups(groups, args.test_fraction, random.Random(seed))
    if test_groups is None:
        return {"status": "skipped_one_group", "n_valid": len(rows), "threshold": threshold}

    result = pd.DataFrame(rows)
    result["group"] = groups
    result["split"] = np.where(result["group"].isin(test_groups), "test", "train")
    train_labels = result.loc[result["split"] == "train", "label"]
    test_labels = result.loc[result["split"] == "test", "label"]
    if len(train_labels) == 0 or len(test_labels) == 0:
        return {"status": "skipped_empty_split", "n_valid": len(rows), "threshold": threshold}

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    result.to_csv(out_path, index=False)
    return {
        "status": "ok",
        "n_valid": len(result),
        "n_groups": len(set(groups)),
        "n_train": int((result["split"] == "train").sum()),
        "n_test": int((result["split"] == "test").sum()),
        "train_label0": int((train_labels == 0).sum()),
        "train_label1": int((train_labels == 1).sum()),
        "test_label0": int((test_labels == 0).sum()),
        "test_label1": int((test_labels == 1).sum()),
        "threshold_mean_value_num": threshold,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate Butina cluster-holdout CSV splits")
    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--csv_name", default="IC50.CSV")
    parser.add_argument("--smiles_col", default="compound_smiles")
    parser.add_argument("--value_col", default="value_num")
    parser.add_argument("--test_fraction", type=float, default=0.30)
    parser.add_argument("--min_molecules", type=int, default=21)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.threshold < 1:
        parser.error("--threshold must be between 0 and 1")

    rows = []
    targets = sorted(
        name for name in os.listdir(args.root_dir)
        if os.path.isdir(os.path.join(args.root_dir, name))
    )
    for index, target in enumerate(targets):
        target_dir = os.path.join(args.root_dir, target)
        matches = [
            name for name in os.listdir(target_dir)
            if name.lower() == args.csv_name.lower()
        ]
        if not matches:
            rows.append({"target": target, "status": "no_csv"})
            continue
        try:
            output_csv = os.path.join(args.output_dir, target, "IC50.holdout.csv")
            row = process_one(
                os.path.join(target_dir, matches[0]), output_csv, args, args.seed + index
            )
            row["target"] = target
            rows.append(row)
            print(f"[{row['status']}] {target} n={row.get('n_valid', 0)}")
        except Exception as exc:
            rows.append({"target": target, "status": "error", "message": str(exc)})
            print(f"[error] {target}: {exc}")

    os.makedirs(args.output_dir, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with open(os.path.join(args.output_dir, "cluster_split_summary.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
