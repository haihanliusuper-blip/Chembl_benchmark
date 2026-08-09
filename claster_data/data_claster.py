#!/usr/bin/env python3
"""Backward-compatible split-only wrapper; prefer run_cluster_pipeline.py."""
import argparse
import subprocess
import sys
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description="Generate deterministic Butina cluster-holdout splits")
    p.add_argument("--root_dir", required=True)
    p.add_argument("--output_dir", default="cluster_benchmark/splits")
    p.add_argument("--csv_name", default="IC50.csv")
    p.add_argument("--smiles_col", default="compound_smiles")
    p.add_argument("--value_col", default="value_num")
    p.add_argument("--threshold", type=float, default=0.6)
    p.add_argument("--cluster_test_frac", type=float, default=0.20)
    p.add_argument("--min_molecules", type=int, default=21)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--python_exec", default=sys.executable)
    a = p.parse_args()
    script = Path(__file__).resolve().parent / "cluster_split_ic50.py"
    cmd = [a.python_exec, str(script), "--root_dir", a.root_dir,
           "--output_dir", a.output_dir, "--csv_name", a.csv_name,
           "--smiles_col", a.smiles_col, "--value_col", a.value_col,
           "--threshold", str(a.threshold), "--test_fraction", str(a.cluster_test_frac),
           "--min_molecules", str(a.min_molecules), "--seed", str(a.seed)]
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()

