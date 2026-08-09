#!/usr/bin/env python3
"""Check required imports, files, and CLI entry points without training."""

import importlib
import subprocess
import sys
from pathlib import Path


REQUIRED_MODULES = [
    "numpy", "pandas", "sklearn", "torch", "torch_geometric", "rdkit",
    "matplotlib", "requests", "tqdm",
]
REQUIRED_FILES = [
    "claster_data/process_ic50_to_pt.py",
    "claster_data/cluster_split_ic50.py",
    "claster_data/cluster_csv_to_pt.py",
    "claster_data/run_cluster_pipeline.py",
    "claster_data/train_claster.py",
    "train/GIN_R.py", "train/GCN.py", "train/GAT.py", "train/GIN_SAGEConv.py",
]


def main():
    root = Path(__file__).resolve().parent
    failed = False
    for module in REQUIRED_MODULES:
        try:
            imported = importlib.import_module(module)
            print(f"[OK import] {module} {getattr(imported, '__version__', '')}")
        except Exception as exc:
            failed = True
            print(f"[FAIL import] {module}: {exc}")
    for relative in REQUIRED_FILES:
        path = root / relative
        if not path.is_file():
            failed = True
            print(f"[MISSING] {relative}")
            continue
        completed = subprocess.run(
            [sys.executable, str(path), "--help"], cwd=root,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        )
        if completed.returncode == 0:
            print(f"[OK CLI] {relative}")
        else:
            failed = True
            print(f"[FAIL CLI] {relative}: {completed.stderr[-300:]}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

