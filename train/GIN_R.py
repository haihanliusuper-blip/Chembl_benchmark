# -*- coding: utf-8 -*-
"""
Batch train GIN over multiple targets (.pt),
save per-target best model weights as <target>.pt,
and save a summary CSV into the same output directory.

Example:
python train_gin_batch_saveweights.py ^
  --root ChEMBL_Targets_MIN ^
  --pt_name IC50_mean.pt ^
  --task bin ^
  --out_dir "E:\\shanda26_1\\01\\反向找靶\\GIN"
"""

import os
import argparse
import time
import csv
import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch.nn import Sequential as Seq, Linear, ReLU
from sklearn.metrics import r2_score, roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split


# ------------------------
# 复现性
# ------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------
# 模型
# ------------------------
class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, dropout_rate=0.5):
        super(GIN, self).__init__()
        self.mlp1 = Seq(Linear(input_dim, hidden_dim1), ReLU(), Linear(hidden_dim1, hidden_dim1))
        self.conv1 = GINConv(self.mlp1)
        self.mlp2 = Seq(Linear(hidden_dim1, hidden_dim2), ReLU(), Linear(hidden_dim2, hidden_dim2))
        self.conv2 = GINConv(self.mlp2)
        self.dropout = nn.Dropout(dropout_rate)
        self.out = Linear(hidden_dim2, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = global_mean_pool(x, batch)
        x = self.out(x)
        return x


# ------------------------
# 工具函数
# ------------------------
def get_labels(data_list):
    ys = []
    for d in data_list:
        y = d.y.view(-1).detach().cpu().numpy()
        if y.size == 0:
            raise ValueError("Found empty y in a Data object.")
        ys.append(y[0])
    return np.array(ys)


@torch.no_grad()
def evaluate(model, loader, task, criterion, device):
    total_loss = 0.0
    all_true, all_pred, all_pred_label = [], [], []

    for data in loader:
        data = data.to(device)
        logits = model(data)

        if task == "reg":
            target = data.y.view(-1).float()
            pred = logits.view(-1).float()
            loss = criterion(pred, target)
            total_loss += float(loss.item())
            all_true.extend(target.detach().cpu().numpy().tolist())
            all_pred.extend(pred.detach().cpu().numpy().tolist())

        elif task == "bin":
            target = data.y.view(-1).float()
            logit = logits.view(-1)
            loss = criterion(logit, target)
            total_loss += float(loss.item())
            prob = torch.sigmoid(logit).detach().cpu().numpy()
            labl = (prob >= 0.5).astype(np.int64)
            all_true.extend(target.detach().cpu().numpy().tolist())
            all_pred.extend(prob.tolist())
            all_pred_label.extend(labl.tolist())

        else:  # multi
            target = data.y.view(-1).long()
            loss = criterion(logits, target)
            total_loss += float(loss.item())
            probs = F.softmax(logits, dim=1).detach().cpu().numpy()
            preds = probs.argmax(axis=1)
            all_true.extend(target.detach().cpu().numpy().tolist())
            all_pred.extend(probs.tolist())
            all_pred_label.extend(preds.tolist())

    metrics = {}
    avg_loss = total_loss / max(1, len(loader))

    if task == "reg":
        r2 = r2_score(all_true, all_pred) if len(all_true) > 1 else float("nan")
        metrics.update({"loss": avg_loss, "r2": r2})

    elif task == "bin":
        acc = accuracy_score(all_true, all_pred_label) if len(all_pred_label) > 0 else float("nan")
        # 这里的宏F1需要至少出现两类才有意义，否则会报/变得奇怪；做个保守处理
        f1m = f1_score(all_true, all_pred_label, average="macro") if len(set(all_true)) > 1 else float("nan")
        try:
            auc = roc_auc_score(all_true, all_pred)
        except Exception:
            auc = float("nan")
        metrics.update({"loss": avg_loss, "acc": acc, "auc": auc, "f1_macro": f1m})

    else:  # multi
        acc = accuracy_score(all_true, all_pred_label) if len(all_pred_label) > 0 else float("nan")
        f1m = f1_score(all_true, all_pred_label, average="macro") if len(set(all_true)) > 1 else float("nan")
        try:
            auc = roc_auc_score(np.array(all_true), np.array(all_pred), multi_class="ovr")
        except Exception:
            auc = float("nan")
        metrics.update({"loss": avg_loss, "acc": acc, "auc": auc, "f1_macro": f1m})

    return metrics


def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def train_one_target(payload_path, target_name, args, device, out_dir):
    # 加载数据
    payload = torch.load(payload_path, map_location="cpu")
    if "data_list" not in payload:
        raise KeyError(f"{payload_path} missing key 'data_list'. Keys={list(payload.keys())}")
    data_list = payload["data_list"]

    if len(data_list) <= args.min_size:
        return {"status": "skipped_size", "n_graphs": len(data_list)}

    labels_np = get_labels(data_list)

    # 任务/输出维度
    task = args.task
    inferred_num_classes = None
    if task == "multi":
        if args.num_classes and args.num_classes > 0:
            num_classes = args.num_classes
        else:
            unique_vals = np.unique(labels_np.astype(int))
            # 若是 0..K-1 连续整数，取 max+1；否则取 unique 数量
            if unique_vals.min() == 0 and np.all(np.diff(np.sort(unique_vals)) == 1):
                num_classes = int(unique_vals.max() + 1)
            else:
                num_classes = int(len(unique_vals))
            inferred_num_classes = num_classes
    else:
        num_classes = None

    # 分类任务：单一类别直接跳过
    if task in ["bin", "multi"]:
        y_int = labels_np.astype(int)
        if len(np.unique(y_int)) < 2:
            label0 = int((y_int == 0).sum())
            label1 = int((y_int == 1).sum())
            return {
                "status": "skipped_single_class",
                "n_graphs": len(data_list),
                "label0": label0,
                "label1": label1,
            }

    # 划分
    indices = np.arange(len(data_list))
    if task in ["bin", "multi"]:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=args.test_ratio,
            random_state=args.seed,
            stratify=labels_np.astype(int),
        )
    else:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=args.test_ratio,
            random_state=args.seed,
            shuffle=True,
        )

    train_data = [data_list[i] for i in train_idx]
    test_data = [data_list[i] for i in test_idx]
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    num_features = data_list[0].num_node_features
    output_dim = 1 if task in ["reg", "bin"] else int(num_classes)
    model = GIN(num_features, args.hidden_dim1, args.hidden_dim2, output_dim, args.dropout_rate).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if task == "reg":
        criterion = nn.MSELoss()
    elif task == "bin":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # 训练：记录 best 权重
    best_score = -float("inf")
    best_state_dict = None
    best_epoch = -1
    metric_key = {"reg": "r2", "bin": "auc", "multi": "acc"}[task]

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            logits = model(data)

            if task == "reg":
                target = data.y.view(-1).float()
                pred = logits.view(-1).float()
                loss = criterion(pred, target)
            elif task == "bin":
                target = data.y.view(-1).float()
                pred = logits.view(-1)
                loss = criterion(pred, target)
            else:
                target = data.y.view(-1).long()
                loss = criterion(logits, target)

            loss.backward()
            optimizer.step()

        # 每轮评估（这里用 test_loader 当作验证集做 best selection）
        metrics = evaluate(model, test_loader, task, criterion, device)
        score = metrics.get(metric_key, float("nan"))
        if np.isnan(score) and task == "bin":
            score = metrics.get("acc", float("nan"))

        if not np.isnan(score) and score > best_score:
            best_score = float(score)
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

    t1 = time.time()

    # 终局评估（用最后一轮参数的结果；另补一份 best 参数的结果）
    final_metrics_last = evaluate(model, test_loader, task, criterion, device)

    best_metrics = None
    if best_state_dict is not None:
        model_best = GIN(num_features, args.hidden_dim1, args.hidden_dim2, output_dim, args.dropout_rate).to(device)
        model_best.load_state_dict(best_state_dict, strict=True)
        best_metrics = evaluate(model_best, test_loader, task, criterion, device)

    # 统计标签分布（整体）
    label0 = int((labels_np == 0).sum()) if task in ["bin", "multi"] else ""
    label1 = int((labels_np == 1).sum()) if task in ["bin", "multi"] else ""

    # 保存 best 权重到 out_dir/<target>.pt
    saved_weight_path = ""
    if best_state_dict is not None:
        safe_mkdir(out_dir)
        saved_weight_path = os.path.join(out_dir, f"{target_name}.pt")

        ckpt = {
            "target": target_name,
            "task": task,
            "num_features": int(num_features),
            "output_dim": int(output_dim),
            "hidden_dim1": int(args.hidden_dim1),
            "hidden_dim2": int(args.hidden_dim2),
            "dropout_rate": float(args.dropout_rate),
            "best_epoch": int(best_epoch),
            "best_score": float(best_score),
            "best_metric_key": metric_key,
            "state_dict": best_state_dict,
            "args": vars(args),
            "split": {
                "n_graphs": int(len(data_list)),
                "n_train": int(len(train_data)),
                "n_test": int(len(test_data)),
                "seed": int(args.seed),
                "test_ratio": float(args.test_ratio),
            },
            "metrics_best_on_test": best_metrics if best_metrics is not None else {},
        }
        torch.save(ckpt, saved_weight_path)

    # 输出行
    # 为了更可读：CSV 里优先写 best 模型在 test 上的指标；没有 best 就回退 last
    use_metrics = best_metrics if best_metrics is not None else final_metrics_last

    row = {
        "status": "ok",
        "n_graphs": len(data_list),
        "n_train": len(train_data),
        "n_test": len(test_data),
        "task": task,
        "loss": float(use_metrics.get("loss", float("nan"))),
        "r2": float(use_metrics.get("r2", float("nan"))) if task == "reg" else "",
        "acc": float(use_metrics.get("acc", float("nan"))) if task in ["bin", "multi"] else "",
        "auc": float(use_metrics.get("auc", float("nan"))) if task in ["bin", "multi"] else "",
        "f1_macro": float(use_metrics.get("f1_macro", float("nan"))) if task in ["bin", "multi"] else "",
        "label0": label0,
        "label1": label1,
        "best_score": float(best_score) if not np.isinf(best_score) else "",
        "best_epoch": int(best_epoch) if best_epoch > 0 else "",
        "secs": round(t1 - t0, 2),
        "weight_path": saved_weight_path,
        "num_classes": inferred_num_classes if task == "multi" else "",
    }
    return row


# ------------------------
# 主程序：批量跑 + 保存权重 + 输出CSV
# ------------------------
def main():
    parser = argparse.ArgumentParser(description="Batch train GIN over multiple targets (.pt), save weights and report")
    parser.add_argument("--root", type=str, default="ChEMBL_Targets_MIN", help="根目录，包含各靶点子文件夹")
    parser.add_argument("--pt_name", type=str, default="IC50_mean.pt", help="每个子文件夹里要加载的 .pt 文件名")

    parser.add_argument(
        "--out_dir",
        type=str,
        default=r"E:\shanda26_1\01\反向找靶\GIN",
        help="输出目录：保存每个target的模型权重与汇总CSV",
    )
    parser.add_argument("--report_csv", type=str, default="report_gin.csv", help="输出报告CSV文件名（放在out_dir里）")

    parser.add_argument("--task", type=str, default="bin", choices=["reg", "bin", "multi"], help="任务类型")
    parser.add_argument("--num_classes", type=int, default=0, help="多分类类别数(0=自动)")
    parser.add_argument("--hidden_dim1", type=int, default=128)
    parser.add_argument("--hidden_dim2", type=int, default=256)
    parser.add_argument("--dropout_rate", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--min_size", type=int, default=20, help="仅当样本数 > min_size 才训练（默认>20）")
    parser.add_argument("--seed", type=int, default=42)

    # 可选：把单靶点权重放到 out_dir/weights 子目录里，避免跟 report 混
    parser.add_argument("--weights_subdir", type=str, default="weights", help="权重文件子目录名")
    parser.add_argument("--save_json_summary", action="store_true", help="额外保存一份 summary.json（放在out_dir里）")

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = args.root
    if not os.path.isdir(root):
        raise FileNotFoundError(f"root not found: {root}")

    out_dir = args.out_dir
    weights_dir = os.path.join(out_dir, args.weights_subdir)
    safe_mkdir(weights_dir)
    safe_mkdir(out_dir)

    subfolders = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])

    rows = []
    for sub in subfolders:
        pt_path = os.path.join(root, sub, args.pt_name)

        if not os.path.isfile(pt_path):
            rows.append({
                "target": sub, "status": "no_pt", "n_graphs": "", "n_train": "", "n_test": "",
                "task": args.task, "loss": "", "r2": "", "acc": "", "auc": "", "f1_macro": "",
                "label0": "", "label1": "", "best_score": "", "best_epoch": "", "secs": "",
                "weight_path": "", "num_classes": ""
            })
            continue

        print(f"\n=== [{sub}] ===")
        try:
            # 注意：权重统一保存到 weights_dir，文件名为 <target>.pt
            row = train_one_target(pt_path, sub, args, device, weights_dir)
            row["target"] = sub
            rows.append(row)
            print(f"[Done] {sub} -> status={row.get('status')} weight={row.get('weight_path')}")
        except Exception as e:
            print(f"[FAIL] {sub}: {e}")
            rows.append({
                "target": sub, "status": f"error:{e}", "n_graphs": "", "n_train": "", "n_test": "",
                "task": args.task, "loss": "", "r2": "", "acc": "", "auc": "", "f1_macro": "",
                "label0": "", "label1": "", "best_score": "", "best_epoch": "", "secs": "",
                "weight_path": "", "num_classes": ""
            })

    # 写CSV到 out_dir
    csv_path = os.path.join(out_dir, args.report_csv)
    fieldnames = [
        "target", "status", "n_graphs", "n_train", "n_test", "task",
        "loss", "r2", "acc", "auc", "f1_macro",
        "label0", "label1", "best_score", "best_epoch", "secs",
        "weight_path", "num_classes"
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            # 防止缺字段
            for k in fieldnames:
                if k not in r:
                    r[k] = ""
            writer.writerow(r)

    print(f"\n✅ 汇总报告已写入: {csv_path}")
    print(f"✅ 权重文件目录: {weights_dir}")

    # 可选：保存json总结
    if args.save_json_summary:
        summary = {
            "root": root,
            "pt_name": args.pt_name,
            "task": args.task,
            "device": str(device),
            "out_dir": out_dir,
            "weights_dir": weights_dir,
            "report_csv": csv_path,
            "args": vars(args),
            "rows": rows,
        }
        json_path = os.path.join(out_dir, "summary.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"✅ summary.json 已写入: {json_path}")


if __name__ == "__main__":
    main()
