#!/usr/bin/env python3
"""Reproducible end-to-end Butina cluster-holdout benchmark."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(command):
    print("\n+", " ".join(map(str, command)), flush=True)
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run split, PyG conversion, and GIN cluster training")
    parser.add_argument("--root_dir", required=True, help="Target directories containing IC50.csv")
    parser.add_argument("--work_dir", default="cluster_benchmark")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--test_fraction", type=float, default=0.20)
    parser.add_argument("--min_molecules", type=int, default=21)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--skip_split", action="store_true")
    parser.add_argument("--skip_conversion", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    root = Path(args.root_dir).resolve()
    work = Path(args.work_dir).resolve()
    splits, pt_root = work / "splits", work / "pt"
    work.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update(root_dir=str(root), work_dir=str(work), python=sys.executable)
    (work / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    if not args.skip_split:
        run([
            sys.executable, str(here / "cluster_split_ic50.py"),
            "--root_dir", str(root), "--output_dir", str(splits),
            "--threshold", str(args.threshold), "--test_fraction", str(args.test_fraction),
            "--min_molecules", str(args.min_molecules), "--seed", str(args.seed),
        ])
    if not args.skip_conversion:
        run([
            sys.executable, str(here / "cluster_csv_to_pt.py"),
            "--input_dir", str(splits), "--output_dir", str(pt_root), "--resume",
        ])
    if not args.skip_train:
        run([
            sys.executable, str(here / "train_claster.py"), "--root", str(pt_root),
            "--train_name", "IC50.holdout.train.pt", "--test_name", "IC50.holdout.test.pt",
            "--report_csv", "report_gin_cluster.csv", "--task", "bin",
            "--epochs", str(args.epochs), "--batch_size", str(args.batch_size),
            "--lr", str(args.lr), "--seed", str(args.seed),
        ])


if __name__ == "__main__":
    main()
