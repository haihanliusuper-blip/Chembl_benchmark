#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Convert cluster-split CSV files into train/test PyG payloads safely.

Each target is converted in an isolated child process. A native-library crash
can therefore fail only one target instead of terminating the full experiment.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile

import pandas as pd
import torch

from process_ic50_to_pt import smiles_to_data


def make_payload(frame, smiles_col, label_col, source_csv):
    data_list, smiles, labels, skipped = [], [], [], 0
    for smi, label in zip(frame[smiles_col].astype(str), frame[label_col]):
        try:
            graph = smiles_to_data(smi, int(label))
        except Exception:
            graph = None
        if graph is None:
            skipped += 1
            continue
        data_list.append(graph)
        smiles.append(smi)
        labels.append(int(label))
    return {
        "data_list": data_list,
        "x_smiles": smiles,
        "labels": labels,
        "source_csv": os.path.abspath(source_csv),
        "skipped_bad_smiles": skipped,
    }


def atomic_torch_save(payload, destination):
    directory = os.path.dirname(destination)
    fd, temporary = tempfile.mkstemp(prefix=".writing_", suffix=".pt", dir=directory)
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def convert_one(args, target):
    csv_path = os.path.join(args.input_dir, target, args.csv_name)
    frame = pd.read_csv(csv_path)
    required = [args.smiles_col, args.label_col, "split"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"missing columns {missing}")
    train = make_payload(frame[frame["split"] == "train"], args.smiles_col, args.label_col, csv_path)
    test = make_payload(frame[frame["split"] == "test"], args.smiles_col, args.label_col, csv_path)
    target_out = os.path.join(args.output_dir, target)
    os.makedirs(target_out, exist_ok=True)
    atomic_torch_save(train, os.path.join(target_out, "IC50.holdout.train.pt"))
    atomic_torch_save(test, os.path.join(target_out, "IC50.holdout.test.pt"))
    return {
        "target": target, "status": "ok", "n_train": len(train["data_list"]),
        "n_test": len(test["data_list"]), "train_label0": train["labels"].count(0),
        "train_label1": train["labels"].count(1), "test_label0": test["labels"].count(0),
        "test_label1": test["labels"].count(1),
    }


def child_main(args):
    try:
        print(json.dumps(convert_one(args, args.target), ensure_ascii=False))
        return 0
    except Exception as exc:
        print(json.dumps({"target": args.target, "status": "error", "message": str(exc)}, ensure_ascii=False))
        return 1


def parent_main(args):
    rows = []
    for target in sorted(os.listdir(args.input_dir)):
        if not os.path.isfile(os.path.join(args.input_dir, target, args.csv_name)):
            continue
        target_out = os.path.join(args.output_dir, target)
        train_path = os.path.join(target_out, "IC50.holdout.train.pt")
        test_path = os.path.join(target_out, "IC50.holdout.test.pt")
        if args.resume and os.path.isfile(train_path) and os.path.isfile(test_path):
            rows.append({"target": target, "status": "existing"})
            print(f"[existing] {target}")
            continue
        command = [
            sys.executable, os.path.abspath(__file__), "--input_dir", args.input_dir,
            "--output_dir", args.output_dir, "--csv_name", args.csv_name,
            "--smiles_col", args.smiles_col, "--label_col", args.label_col,
            "--target", target,
        ]
        completed = subprocess.run(command, text=True, capture_output=True)
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        try:
            row = json.loads(lines[-1])
        except (IndexError, json.JSONDecodeError):
            row = {
                "target": target,
                "status": "child_crash" if completed.returncode < 0 else "child_error",
                "message": f"exit={completed.returncode}; stderr={completed.stderr[-500:]}",
            }
        rows.append(row)
        print(f"[{row['status']}] {target}: train={row.get('n_train', '')} test={row.get('n_test', '')}")

    os.makedirs(args.output_dir, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with open(os.path.join(args.output_dir, "pt_conversion_summary.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Convert cluster-split CSVs to train/test .pt files")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--csv_name", default="IC50.holdout.csv")
    parser.add_argument("--smiles_col", default="compound_smiles")
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--resume", action="store_true", help="Skip targets with both output .pt files")
    parser.add_argument("--target", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    return child_main(args) if args.target else parent_main(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
