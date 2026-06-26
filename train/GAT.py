# -*- coding: utf-8 -*-
import os
import argparse
import time
import csv
import random
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

# ------------------------
class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, dropout_rate=0.5):
        super(GIN, self).__init__()
        # 用 GATConv 代替原先的 GINConv；保持输出维度与 out_channels 一致
        self.conv1 = GATConv(
            in_channels=input_dim,
            out_channels=hidden_dim1,
            heads=4,
            concat=False,
            dropout=dropout_rate
        )
        self.conv2 = GATConv(
            in_channels=hidden_dim1,
            out_channels=hidden_dim2,
            heads=4,
            concat=False,
            dropout=dropout_rate
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.out = nn.Linear(hidden_dim2, output_dim)

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
def get_labels(dl):
    ys = []
    for d in dl:
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
            total_loss += loss.item()
            all_true.extend(target.detach().cpu().numpy().tolist())
            all_pred.extend(pred.detach().cpu().numpy().tolist())
        elif task == "bin":
            target = data.y.view(-1).float()
            logit = logits.view(-1)
            loss = criterion(logit, target)
            total_loss += loss.item()
            prob = torch.sigmoid(logit).detach().cpu().numpy()
            labl = (prob >= 0.5).astype(np.int64)
            all_true.extend(target.detach().cpu().numpy().tolist())
            all_pred.extend(prob.tolist())
            all_pred_label.extend(labl.tolist())
        else:  # multi
            target = data.y.view(-1).long()
            loss = criterion(logits, target)
            total_loss += loss.item()
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
        acc = accuracy_score(all_true, all_pred_label) if len(set(all_pred_label)) > 0 else float("nan")
        f1m = f1_score(all_true, all_pred_label, average="macro") if len(set(all_pred_label)) > 1 else float("nan")
        try:
            auc = roc_auc_score(all_true, all_pred)
        except Exception:
            auc = float("nan")
        metrics.update({"loss": avg_loss, "acc": acc, "auc": auc, "f1_macro": f1m})
    else:
        acc = accuracy_score(all_true, all_pred_label) if len(set(all_pred_label)) > 0 else float("nan")
        f1m = f1_score(all_true, all_pred_label, average="macro") if len(set(all_pred_label)) > 1 else float("nan")
        try:
            auc = roc_auc_score(np.array(all_true), np.array(all_pred), multi_class="ovr")
        except Exception:
            auc = float("nan")
        metrics.update({"loss": avg_loss, "acc": acc, "auc": auc, "f1_macro": f1m})
    return metrics

def train_one_target(payload_path, args, device):
    # 加载数据
    payload = torch.load(payload_path, map_location="cpu")
    data_list = payload["data_list"]
    if len(data_list) <= args.min_size:
        return {"status": "skipped_size", "n_graphs": len(data_list)}

    labels_np = get_labels(data_list)

    # 任务/输出维度
    task = args.task
    if task == "multi" and (args.num_classes is None or args.num_classes <= 0):
        unique_vals = np.unique(labels_np.astype(int))
        num_classes = int(unique_vals.max() + 1) if (unique_vals.min() == 0 and np.all(np.diff(np.sort(unique_vals)) == 1)) else len(unique_vals)
    else:
        num_classes = args.num_classes

    if task in ["bin", "multi"]:
        y_int = labels_np.astype(int)
        # 单一类别直接跳过（无法分层/无法训练出有意义分类器）
        if len(np.unique(y_int)) < 2:
            return {"status": "skipped_single_class", "n_graphs": len(data_list), "label0": int((y_int == 0).sum()), "label1": int((y_int == 1).sum())}

    # 划分
    indices = np.arange(len(data_list))
    if task in ["bin", "multi"]:
        train_idx, test_idx = train_test_split(indices, test_size=args.test_ratio, random_state=args.seed, stratify=labels_np.astype(int))
    else:
        train_idx, test_idx = train_test_split(indices, test_size=args.test_ratio, random_state=args.seed, shuffle=True)

    train_data = [data_list[i] for i in train_idx]
    test_data  = [data_list[i] for i in test_idx]
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_data,  batch_size=args.batch_size, shuffle=False)

    num_features = data_list[0].num_node_features
    output_dim = 1 if task in ["reg", "bin"] else num_classes
    model = GIN(num_features, args.hidden_dim1, args.hidden_dim2, output_dim, args.dropout_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss() if task == "reg" else (nn.BCEWithLogitsLoss() if task == "bin" else nn.CrossEntropyLoss())

    # 训练
    best_score = -float("inf")
    metric_key = {"reg":"r2","bin":"auc","multi":"acc"}[task]
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
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
                pred = logits
                loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 每轮评估（不保存权重）
        metrics = evaluate(model, test_loader, task, criterion, device)
        score = metrics.get(metric_key, float("nan"))
        if np.isnan(score):
            # 对于二分类若AUC为NaN，则用ACC作为回退
            if task == "bin":
                score = metrics.get("acc", float("nan"))
        if not np.isnan(score) and score > best_score:
            best_score = score

    t1 = time.time()
    # 终局评估
    final_metrics = evaluate(model, test_loader, task, criterion, device)

    # 统计标签分布（整体）
    label0 = int((labels_np == 0).sum()) if task in ["bin", "multi"] else ""
    label1 = int((labels_np == 1).sum()) if task in ["bin", "multi"] else ""

    return {
        "status": "ok",
        "n_graphs": len(data_list),
        "n_train": len(train_data),
        "n_test": len(test_data),
        "task": task,
        "loss": float(final_metrics.get("loss", float("nan"))),
        "r2": float(final_metrics.get("r2", float("nan"))) if task == "reg" else "",
        "acc": float(final_metrics.get("acc", float("nan"))) if task in ["bin","multi"] else "",
        "auc": float(final_metrics.get("auc", float("nan"))) if task in ["bin","multi"] else "",
        "f1_macro": float(final_metrics.get("f1_macro", float("nan"))) if task in ["bin","multi"] else "",
        "label0": label0,
        "label1": label1,
        "best_score": float(best_score) if not np.isinf(best_score) else "",
        "secs": round(t1 - t0, 2),
    }

# ------------------------
# 主程序：批量跑
# ------------------------
def main():
    parser = argparse.ArgumentParser(description="Batch train GIN over multiple targets (.pt) and write report CSV")
    parser.add_argument("--root", type=str, default="ChEMBL_Targets_MIN", help="根目录，包含各靶点子文件夹")
    parser.add_argument("--pt_name", type=str, default="IC50_mean.pt", help="每个子文件夹里要加载的 .pt 文件名")
    parser.add_argument("--report_csv", type=str, default="report_gcn.csv", help="输出报告文件名")
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
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = []
    root = args.root
    subfolders = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])

    for sub in subfolders:
        pt_path = os.path.join(root, sub, args.pt_name)
        if not os.path.isfile(pt_path):
            # 没有 pt 就跳过
            rows.append({
                "target": sub, "status": "no_pt", "n_graphs": "", "n_train": "", "n_test": "",
                "task": args.task, "loss": "", "r2": "", "acc": "", "auc": "", "f1_macro": "",
                "label0": "", "label1": "", "best_score": "", "secs": ""
            })
            continue

        print(f"\n=== [{sub}] ===")
        try:
            result = train_one_target(pt_path, args, device)
            row = {
                "target": sub,
                "status": result.get("status", "ok"),
                "n_graphs": result.get("n_graphs", ""),
                "n_train": result.get("n_train", ""),
                "n_test":  result.get("n_test", ""),
                "task":    result.get("task", args.task),
                "loss":    result.get("loss", ""),
                "r2":      result.get("r2", ""),
                "acc":     result.get("acc", ""),
                "auc":     result.get("auc", ""),
                "f1_macro":result.get("f1_macro", ""),
                "label0":  result.get("label0", ""),
                "label1":  result.get("label1", ""),
                "best_score": result.get("best_score", ""),
                "secs":    result.get("secs", "")
            }
            rows.append(row)
            print(f"[Done] {sub} -> {row}")
        except Exception as e:
            print(f"[FAIL] {sub}: {e}")
            rows.append({
                "target": sub, "status": f"error:{e}", "n_graphs": "", "n_train": "", "n_test": "",
                "task": args.task, "loss": "", "r2": "", "acc": "", "auc": "", "f1_macro": "",
                "label0": "", "label1": "", "best_score": "", "secs": ""
            })

    # 写CSV
    csv_path = os.path.join(root, args.report_csv)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "target","status","n_graphs","n_train","n_test","task",
            "loss","r2","acc","auc","f1_macro","label0","label1","best_score","secs"
        ])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"\n✅ 报告已写入: {csv_path}")

if __name__ == "__main__":
    main()
