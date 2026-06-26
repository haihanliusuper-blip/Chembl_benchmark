# -*- coding: utf-8 -*-
import os
import csv
import time
import random
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_pt_payload(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class GINEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, dropout_rate=0.5):
        super().__init__()

        self.mlp1 = Seq(
            Linear(input_dim, hidden_dim1),
            ReLU(),
            Linear(hidden_dim1, hidden_dim1)
        )
        self.conv1 = GINConv(self.mlp1)

        self.mlp2 = Seq(
            Linear(hidden_dim1, hidden_dim2),
            ReLU(),
            Linear(hidden_dim2, hidden_dim2)
        )
        self.conv2 = GINConv(self.mlp2)

        self.dropout = nn.Dropout(dropout_rate)
        self.out_dim = hidden_dim2

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)

        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)

        g = global_mean_pool(x, batch)
        return g


class GINMultiTaskBin(nn.Module):
    """
    共享 GIN encoder + 每个 task 一个二分类 head
    """
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, num_tasks, dropout_rate=0.5):
        super().__init__()

        self.encoder = GINEncoder(
            input_dim=input_dim,
            hidden_dim1=hidden_dim1,
            hidden_dim2=hidden_dim2,
            dropout_rate=dropout_rate
        )

        self.heads = nn.ModuleList([
            Linear(hidden_dim2, 1) for _ in range(num_tasks)
        ])

    def forward(self, data):
        g = self.encoder(data)
        task_idx = data.task_idx.view(-1).long()

        logits = torch.zeros(g.size(0), device=g.device, dtype=g.dtype)

        unique_task_ids = torch.unique(task_idx)
        for tid in unique_task_ids.tolist():
            mask = task_idx == tid
            logits_task = self.heads[tid](g[mask]).view(-1)
            logits[mask] = logits_task

        return logits


def get_labels_from_data_list(data_list):
    ys = []
    for d in data_list:
        y = d.y.view(-1).detach().cpu().numpy()
        if y.size == 0:
            raise ValueError("Found empty y in a Data object.")
        ys.append(y[0])
    return np.array(ys)


def safe_roc_auc(y_true, y_score):
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return roc_auc_score(y_true, y_score)
    except Exception:
        return float("nan")


def safe_acc(y_true, y_pred):
    try:
        return accuracy_score(y_true, y_pred)
    except Exception:
        return float("nan")


def safe_f1_macro(y_true, y_pred):
    try:
        if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
            return float("nan")
        return f1_score(y_true, y_pred, average="macro")
    except Exception:
        return float("nan")


def ensure_same_feature_dim(task_to_data):
    dims = {}

    for task_name, data_list in task_to_data.items():
        if len(data_list) == 0:
            continue
        dims[task_name] = data_list[0].num_node_features

    uniq_dims = sorted(set(dims.values()))

    if len(uniq_dims) != 1:
        detail = ", ".join([f"{k}:{v}" for k, v in dims.items()])
        raise ValueError(
            f"Different num_node_features across tasks, cannot do shared multitask training. "
            f"Details: {detail}"
        )

    return uniq_dims[0]


def load_small_tasks_for_multitask(root, pt_name, min_size, mt_max_size):
    task_to_data = {}
    skipped_info = {}

    subfolders = sorted([
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ])

    for sub in subfolders:
        pt_path = os.path.join(root, sub, pt_name)

        if not os.path.isfile(pt_path):
            skipped_info[sub] = {"status": "no_pt"}
            continue

        try:
            payload = load_pt_payload(pt_path)
            data_list = payload["data_list"]
        except Exception as e:
            skipped_info[sub] = {"status": f"error_load:{e}"}
            continue

        n = len(data_list)

        if n <= min_size:
            skipped_info[sub] = {
                "status": "skipped_too_small",
                "n_graphs": n
            }
            continue

        if n >= mt_max_size:
            skipped_info[sub] = {
                "status": "skipped_too_large",
                "n_graphs": n
            }
            continue

        labels_np = get_labels_from_data_list(data_list).astype(int)
        uniq = np.unique(labels_np)

        if len(uniq) < 2:
            skipped_info[sub] = {
                "status": "skipped_single_class",
                "n_graphs": n,
                "label0": int((labels_np == 0).sum()),
                "label1": int((labels_np == 1).sum())
            }
            continue

        task_to_data[sub] = data_list

    return task_to_data, skipped_info


def split_small_tasks(task_to_data, test_ratio=0.2, seed=42):
    task_splits = {}
    skipped_info = {}

    for task_name, data_list in task_to_data.items():
        labels = get_labels_from_data_list(data_list).astype(int)
        idx = np.arange(len(data_list))

        uniq, counts = np.unique(labels, return_counts=True)
        n_total = len(data_list)
        n_classes = len(uniq)

        label0 = int((labels == 0).sum())
        label1 = int((labels == 1).sum())

        min_class_count = int(counts.min())
        n_test = max(1, int(round(n_total * test_ratio)))

        if min_class_count < 2:
            skipped_info[task_name] = {
                "status": "skipped_split_min_class_lt2",
                "n_graphs": n_total,
                "label0": label0,
                "label1": label1
            }
            continue

        if n_test < n_classes:
            skipped_info[task_name] = {
                "status": "skipped_split_test_too_small",
                "n_graphs": n_total,
                "label0": label0,
                "label1": label1
            }
            continue

        try:
            train_idx, test_idx = train_test_split(
                idx,
                test_size=test_ratio,
                random_state=seed,
                stratify=labels
            )
        except ValueError as e:
            skipped_info[task_name] = {
                "status": f"skipped_split_error:{e}",
                "n_graphs": n_total,
                "label0": label0,
                "label1": label1
            }
            continue

        task_splits[task_name] = {
            "train": [data_list[i] for i in train_idx],
            "test": [data_list[i] for i in test_idx],
            "labels": labels
        }

    return task_splits, skipped_info


def attach_task_info_to_data_list(task_splits):
    task_names = sorted(task_splits.keys())
    task_to_idx = {t: i for i, t in enumerate(task_names)}

    train_all, test_all = [], []

    for task_name in task_names:
        tid = task_to_idx[task_name]

        for d in task_splits[task_name]["train"]:
            d2 = d.clone()
            d2.task_idx = torch.tensor([tid], dtype=torch.long)
            train_all.append(d2)

        for d in task_splits[task_name]["test"]:
            d2 = d.clone()
            d2.task_idx = torch.tensor([tid], dtype=torch.long)
            test_all.append(d2)

    return train_all, test_all, task_names, task_to_idx


@torch.no_grad()
def evaluate_multitask_bin(model, loader, device, criterion, task_names):
    model.eval()

    total_loss = 0.0
    by_task_true = defaultdict(list)
    by_task_prob = defaultdict(list)
    by_task_pred = defaultdict(list)

    for data in loader:
        data = data.to(device)

        logits = model(data).view(-1)
        target = data.y.view(-1).float()

        loss = criterion(logits, target)
        total_loss += loss.item()

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        preds = (probs >= 0.5).astype(np.int64)
        y_true = target.detach().cpu().numpy().astype(np.int64)
        task_ids = data.task_idx.view(-1).detach().cpu().numpy()

        for i in range(len(y_true)):
            task_name = task_names[int(task_ids[i])]
            by_task_true[task_name].append(int(y_true[i]))
            by_task_prob[task_name].append(float(probs[i]))
            by_task_pred[task_name].append(int(preds[i]))

    task_metrics = {}

    for task_name in sorted(by_task_true.keys()):
        y_true = by_task_true[task_name]
        y_prob = by_task_prob[task_name]
        y_pred = by_task_pred[task_name]

        task_metrics[task_name] = {
            "acc": safe_acc(y_true, y_pred),
            "auc": safe_roc_auc(y_true, y_prob),
            "f1_macro": safe_f1_macro(y_true, y_pred),
            "n_test": len(y_true)
        }

    avg_loss = total_loss / max(1, len(loader))
    return avg_loss, task_metrics


def save_best_encoder_checkpoint(
    save_path,
    model,
    input_dim,
    task_names,
    task_to_idx,
    args,
    best_mean_auc,
    epoch
):
    ckpt = {
        "encoder_state_dict": model.encoder.state_dict(),
        "input_dim": int(input_dim),
        "hidden_dim1": int(args.hidden_dim1),
        "hidden_dim2": int(args.hidden_dim2),
        "dropout_rate": float(args.dropout_rate),
        "best_mean_auc": float(best_mean_auc),
        "best_epoch": int(epoch),
        "task_names": list(task_names),
        "task_to_idx": dict(task_to_idx),
        "args": vars(args),
        "note": "Only shared GIN encoder weights are saved. Task-specific heads are not saved."
    }

    torch.save(ckpt, save_path)


def train_multitask_small_bin(task_to_data, args, device):
    if len(task_to_data) == 0:
        return {}, {
            "status": "no_small_tasks",
            "n_tasks": 0,
            "task_names": [],
            "split_skipped_info": {},
            "encoder_ckpt_path": ""
        }

    input_dim = ensure_same_feature_dim(task_to_data)

    task_splits, split_skipped_info = split_small_tasks(
        task_to_data,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

    if len(task_splits) == 0:
        return {}, {
            "status": "no_tasks_after_split",
            "n_tasks": 0,
            "task_names": [],
            "split_skipped_info": split_skipped_info,
            "encoder_ckpt_path": ""
        }

    train_all, test_all, task_names, task_to_idx = attach_task_info_to_data_list(task_splits)

    train_loader = DataLoader(
        train_all,
        batch_size=args.batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_all,
        batch_size=args.batch_size,
        shuffle=False
    )

    model = GINMultiTaskBin(
        input_dim=input_dim,
        hidden_dim1=args.hidden_dim1,
        hidden_dim2=args.hidden_dim2,
        num_tasks=len(task_names),
        dropout_rate=args.dropout_rate
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    best_mean_score = -float("inf")
    best_epoch = -1

    out_dir = os.path.join(args.root, "多任务_out")
    os.makedirs(out_dir, exist_ok=True)

    encoder_ckpt_path = os.path.join(out_dir, args.encoder_ckpt_name)

    t0 = time.time()

    print("\n[INFO] multitask tasks:")
    for t in task_names:
        print(
            f"  - {t}: total={len(task_to_data[t])}, "
            f"train={len(task_splits[t]['train'])}, "
            f"test={len(task_splits[t]['test'])}"
        )

    if len(split_skipped_info) > 0:
        print("\n[INFO] tasks skipped during split:")
        for t, info in split_skipped_info.items():
            print(f"  - {t}: {info['status']}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0

        for data in train_loader:
            data = data.to(device)

            optimizer.zero_grad()

            logits = model(data).view(-1)
            target = data.y.view(-1).float()

            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        _, test_task_metrics = evaluate_multitask_bin(
            model=model,
            loader=test_loader,
            device=device,
            criterion=criterion,
            task_names=task_names
        )

        aucs = []
        for t in task_names:
            auc = test_task_metrics.get(t, {}).get("auc", float("nan"))
            if not np.isnan(auc):
                aucs.append(auc)

        mean_score = float(np.mean(aucs)) if len(aucs) > 0 else float("nan")

        if not np.isnan(mean_score) and mean_score > best_mean_score:
            best_mean_score = mean_score
            best_epoch = epoch

            save_best_encoder_checkpoint(
                save_path=encoder_ckpt_path,
                model=model,
                input_dim=input_dim,
                task_names=task_names,
                task_to_idx=task_to_idx,
                args=args,
                best_mean_auc=best_mean_score,
                epoch=epoch
            )

            print(
                f"[SAVE] epoch={epoch:03d} "
                f"best_mean_auc={best_mean_score:.4f} "
                f"encoder={encoder_ckpt_path}"
            )

        train_loss_avg = total_train_loss / max(1, len(train_loader))

        if np.isnan(mean_score):
            print(f"[Epoch {epoch:03d}] train_loss={train_loss_avg:.4f} mean_auc=nan")
        else:
            print(f"[Epoch {epoch:03d}] train_loss={train_loss_avg:.4f} mean_auc={mean_score:.4f}")

    t1 = time.time()

    final_loss, final_task_metrics = evaluate_multitask_bin(
        model=model,
        loader=test_loader,
        device=device,
        criterion=criterion,
        task_names=task_names
    )

    rows = {}

    for task_name in task_names:
        labels_np = get_labels_from_data_list(task_to_data[task_name]).astype(int)

        n_total = len(task_to_data[task_name])
        n_train = len(task_splits[task_name]["train"])
        n_test = len(task_splits[task_name]["test"])

        tm = final_task_metrics.get(task_name, {})

        rows[task_name] = {
            "status": "ok_multitask",
            "n_graphs": n_total,
            "n_train": n_train,
            "n_test": n_test,
            "task": "bin",
            "loss": float(final_loss),
            "acc": float(tm.get("acc", float("nan"))),
            "auc": float(tm.get("auc", float("nan"))),
            "f1_macro": float(tm.get("f1_macro", float("nan"))),
            "label0": int((labels_np == 0).sum()),
            "label1": int((labels_np == 1).sum()),
            "best_score": float(best_mean_score) if not np.isinf(best_mean_score) else "",
            "best_epoch": best_epoch,
            "encoder_ckpt": encoder_ckpt_path if os.path.isfile(encoder_ckpt_path) else "",
            "secs": round(t1 - t0, 2),
        }

    return rows, {
        "status": "ok",
        "n_tasks": len(task_names),
        "task_names": task_names,
        "split_skipped_info": split_skipped_info,
        "encoder_ckpt_path": encoder_ckpt_path if os.path.isfile(encoder_ckpt_path) else "",
        "best_epoch": best_epoch,
        "best_mean_auc": best_mean_score
    }


def main():
    parser = argparse.ArgumentParser(
        description="Multitask GIN for small binary tasks, saving only the best shared encoder."
    )

    parser.add_argument("--root", type=str, default="ChEMBL_Targets_MIN")
    parser.add_argument("--pt_name", type=str, default="IC50_mean.pt")
    parser.add_argument("--report_csv", type=str, default="report_gin_multitask_only.csv")

    parser.add_argument("--encoder_ckpt_name", type=str, default="gin_multitask_encoder_best.pt")

    parser.add_argument("--hidden_dim1", type=int, default=128)
    parser.add_argument("--hidden_dim2", type=int, default=256)
    parser.add_argument("--dropout_rate", type=float, default=0.5)

    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--test_ratio", type=float, default=0.5)
    parser.add_argument("--min_size", type=int, default=20, help="样本数 <= min_size 跳过")
    parser.add_argument("--mt_max_size", type=int, default=100, help="样本数 >= mt_max_size 跳过")

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    task_to_data, skipped_info = load_small_tasks_for_multitask(
        root=args.root,
        pt_name=args.pt_name,
        min_size=args.min_size,
        mt_max_size=args.mt_max_size
    )

    multitask_rows, meta_info = train_multitask_small_bin(
        task_to_data=task_to_data,
        args=args,
        device=device
    )

    split_skipped_info = meta_info.get("split_skipped_info", {})

    if meta_info["n_tasks"] == 0:
        print("\n[INFO] 没有最终进入多任务训练的任务。")
    else:
        print(f"\n[INFO] 多任务训练完成，共 {meta_info['n_tasks']} 个任务。")
        print(f"[INFO] best_epoch = {meta_info.get('best_epoch', '')}")
        print(f"[INFO] best_mean_auc = {meta_info.get('best_mean_auc', '')}")
        print(f"[INFO] encoder checkpoint = {meta_info.get('encoder_ckpt_path', '')}")

    rows = []

    all_subfolders = sorted([
        d for d in os.listdir(args.root)
        if os.path.isdir(os.path.join(args.root, d))
    ])

    for sub in all_subfolders:
        if sub in multitask_rows:
            row = {"target": sub}
            row.update(multitask_rows[sub])
        else:
            info = skipped_info.get(sub, {})

            if sub in split_skipped_info:
                info = split_skipped_info[sub]

            row = {
                "target": sub,
                "status": info.get("status", "not_in_multitask"),
                "n_graphs": info.get("n_graphs", ""),
                "n_train": "",
                "n_test": "",
                "task": "bin",
                "loss": "",
                "acc": "",
                "auc": "",
                "f1_macro": "",
                "label0": info.get("label0", ""),
                "label1": info.get("label1", ""),
                "best_score": "",
                "best_epoch": "",
                "encoder_ckpt": "",
                "secs": ""
            }

        rows.append(row)

    csv_path = os.path.join(args.root, args.report_csv)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "target",
                "status",
                "n_graphs",
                "n_train",
                "n_test",
                "task",
                "loss",
                "acc",
                "auc",
                "f1_macro",
                "label0",
                "label1",
                "best_score",
                "best_epoch",
                "encoder_ckpt",
                "secs"
            ]
        )

        writer.writeheader()

        for r in rows:
            writer.writerow(r)

    print(f"\n[SAVED] report: {csv_path}")

    if meta_info.get("encoder_ckpt_path", ""):
        print(f"[SAVED] best encoder: {meta_info['encoder_ckpt_path']}")


if __name__ == "__main__":
    main()