# ChEMBL target-wise GNN Systematic Evaluation

Code accompanying the systematic evaluation of graph neural networks for ligand-based virtual screening on ChEMBL-derived datasets. The repository provides data processing utilities, IC50 preprocessing workflows, target-wise dataset construction, random and scaffold/cluster-based split evaluation, GNN model implementations, property analysis, and multitask learning pipelines.

## Reproducibility status

The reproducibility files requested during peer review are included:

- `environment.yml` and `requirements.txt` specify the software environment;
- `claster_data/cluster_split_ic50.py` implements the missing Butina clustering step;
- `claster_data/cluster_csv_to_pt.py` converts cluster-split CSV files into PyG payloads;
- `claster_data/run_cluster_pipeline.py` is the supported end-to-end cluster workflow;
- command-line help, input schema, random seeds, and expected outputs are documented below.

All random-split and cluster-split commands use explicit seeds. GPU kernels may
still show small platform-dependent numerical differences.

## Repository layout

```text
down_load_data/       ChEMBL activity and UniProt sequence download utilities
claster_data/         IC50 preprocessing and Butina cluster-holdout workflow
train/                GIN, GCN, GAT, and GraphSAGE single-target models
multitask/            multitask GIN training
analyse_property/     target/property summary scripts
environment.yml       recommended Conda environment
requirements.txt      pip-oriented dependency list
```

The historical directory names are retained to avoid breaking published scripts
(`claster_data` means cluster data).

## Environment

The reference environment used to validate the code was:

| Package | Version |
|---|---:|
| Python | 3.9.23 |
| NumPy | 1.24.4 |
| pandas | 2.2.3 |
| scikit-learn | 1.6.1 |
| PyTorch | 2.0.1 |
| PyTorch Geometric | 2.3.1 |
| RDKit | 2024.09.5 |
| matplotlib | 3.8.4 |

Create the environment with:

```bash
conda env create -f environment.yml
conda activate chembl-benchmark
```

PyTorch/CUDA wheels are platform specific. If the solver cannot reproduce the
listed CUDA build, install PyTorch and PyG using the official instructions for
the local CUDA driver, then install the remaining packages from
`requirements.txt`.

Verify imports and command-line entry points:

```bash
python reproducibility_check.py
```

## Input data

Large ChEMBL-derived data files are not stored in Git. For an exact manuscript
reproduction, obtain the archived dataset associated with the article and place
it at `ChEMBL_Targets_MIN/`. Running the downloader against the live ChEMBL API
may return a newer database snapshot and is therefore not guaranteed to recreate
the exact manuscript dataset.

Each target must have its own directory:

```text
ChEMBL_Targets_MIN/
└── <target_name>__<CHEMBL_ID>/
    └── IC50.csv
```

Required `IC50.csv` columns:

| Column | Meaning |
|---|---|
| `compound_smiles` | molecular SMILES used to build the graph |
| `value_num` | numeric IC50 value |
| `value_units` | activity unit |

The download utilities additionally produce `compound_chembl_id`,
`value_relation`, `pchembl_value`, `value_type`, `target_name`, and `target_id`.

To create a current ChEMBL snapshot rather than reproduce the archived one:

```bash
python down_load_data/download_compound.py
python down_load_data/download_protein.py
```

These commands require network access and `chembl_webresource_client`.

## Random-split benchmark

Generate a graph payload in every target directory:

```bash
python claster_data/process_ic50_to_pt.py \
  --root ChEMBL_Targets_MIN \
  --csv_name IC50.csv \
  --summary_csv IC50_mean_summary.csv
```

The original labeling rule is target specific: the arithmetic mean of
`value_num` is calculated independently for each target, and values above that
mean receive label 1. The generated file is `IC50_mean.pt`.

Run the four models with the manuscript defaults (`seed=42`, 80/20 stratified
random split, 100 epochs, batch size 64, learning rate 1e-3):

```bash
python train/GIN_R.py       --root ChEMBL_Targets_MIN --pt_name IC50_mean.pt --task bin --report_csv report_gin.csv
python train/GCN.py         --root ChEMBL_Targets_MIN --pt_name IC50_mean.pt --task bin --report_csv report_gcn.csv
python train/GAT.py         --root ChEMBL_Targets_MIN --pt_name IC50_mean.pt --task bin --report_csv report_gat.csv
python train/GIN_SAGEConv.py --root ChEMBL_Targets_MIN --pt_name IC50_mean.pt --task bin --report_csv report_sage.csv
```

Reports are written inside the directory supplied to `--root`.

## Butina cluster-holdout benchmark

The supported end-to-end entry point is:

```bash
python claster_data/run_cluster_pipeline.py \
  --root_dir ChEMBL_Targets_MIN \
  --work_dir cluster_benchmark \
  --threshold 0.6 \
  --test_fraction 0.20 \
  --seed 42
```

It performs three reproducible steps:

1. Morgan fingerprints (`radius=2`, `2048 bits`) and Butina clustering at the
   requested Tanimoto threshold;
2. whole-cluster test-set selection (no cluster is shared by train and test),
   followed by safe per-target conversion to PyG files;
3. GIN training on the fixed cluster train/test files.

Important outputs:

```text
cluster_benchmark/splits/<target>/IC50.holdout.csv
cluster_benchmark/splits/cluster_split_summary.csv
cluster_benchmark/pt/<target>/IC50.holdout.train.pt
cluster_benchmark/pt/<target>/IC50.holdout.test.pt
cluster_benchmark/pt/pt_conversion_summary.csv
cluster_benchmark/pt/report_gin_cluster.csv
cluster_benchmark/run_config.json
```

Run only split generation for a quick check:

```bash
python claster_data/run_cluster_pipeline.py \
  --root_dir ChEMBL_Targets_MIN \
  --work_dir cluster_smoke \
  --threshold 0.6 --test_fraction 0.20 --skip_train
```

Threshold sensitivity can be reproduced by repeating the command with
`--threshold 0.5`, `0.6`, and `0.7` and separate `--work_dir` values.

## Seeds and split definitions

- Random-split models: default seed `42`.
- Butina cluster holdout: default seed `42`; target index is added to the seed
  so every target has a deterministic but distinct cluster selection.
- Cluster membership is determined before test selection. Complete clusters,
  rather than individual molecules, are assigned to the test set.
- The target activity threshold is calculated before splitting.

## Hardware

Training automatically uses CUDA when available and otherwise runs on CPU.
Preprocessing and Butina clustering are CPU operations. Memory use for Butina
clustering grows quadratically with the number of molecules in an individual
target because pairwise fingerprint distances are required.

## Notes on generated files

Model weights (`*.pt`), downloaded data, reports, logs, and temporary split
directories are ignored by `.gitignore`. The small pretrained file already in
`multitask/` is retained because it was part of the original repository.

