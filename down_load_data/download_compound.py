# -*- coding: utf-8 -*-
"""
chembl_ligand_crawler.py
---------------------------------
功能：
- 扫描 ChEMBL 靶点（target_chembl_id, pref_name）
- 按白名单活性类型拉取（IC50/EC50/Kd/Ki/Inhibition），边拉边写到 CSV
- 自动补充 SMILES
- 目录结构：chembl-ligand/<TargetName>__<CHEMBL_ID>/{IC50.csv, Kd.csv, ..., ALL.csv}
- 断点续传：chembl-ligand/_state.json 中记录当前靶点与 offset
- 已下载合并：读取历史 ALL.csv/各类型 CSV 做去重，避免重复写
- 完成标记：每个靶点目录写 _complete.flag；启动时自动采纳历史完成目录

依赖：
    pip install chembl_webresource_client pandas tqdm
"""

import os
import re
import json
import glob
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from tqdm import tqdm
from chembl_webresource_client.new_client import new_client

# ========== 基本配置 ==========
WHITE_LIST = {"IC50", "EC50", "Kd", "Ki", "Inhibition"}
BASE_DIR = "chembl-ligand"                 # 输出根目录
STATE_PATH = os.path.join(BASE_DIR, "_state.json")
BATCH_SIZE = 500                           # 活性数据抓取批大小
MAX_RECORDS_PER_TARGET = 20000             # 单靶点最大抓取条数，防止爆量
TARGET_LIMIT = 20000               # 总共抓多少个靶点
CHECKPOINT_EVERY_BATCH = 1                 # 每处理多少批保存一次状态

# ========== 小工具 ==========
def safe_filename(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_-]', "_", str(name)).strip("_")[:80] or "Unknown"

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_state() -> dict:
    if not os.path.exists(STATE_PATH):
        return {"done_targets": [], "current": None}
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # 状态损坏：备份并重建
        try:
            os.replace(STATE_PATH, STATE_PATH + ".broken")
        except Exception:
            pass
        return {"done_targets": [], "current": None}

def save_state(state: dict):
    ensure_dir(BASE_DIR)
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, STATE_PATH)

def set_current_target(state: dict, target_id: str, offset: int = 0):
    state["current"] = {"target_id": target_id, "offset": offset}
    save_state(state)

def mark_target_done(state: dict, target_id: str, target_dir: str = None):
    done = set(state.get("done_targets", []))
    done.add(target_id)
    state["done_targets"] = sorted(done)
    state["current"] = None
    save_state(state)
    if target_dir:
        try:
            with open(os.path.join(target_dir, "_complete.flag"), "w", encoding="utf-8") as f:
                f.write("done\n")
        except Exception:
            pass

def build_existing_keys(target_dir: str) -> set:
    """
    扫描 target_dir 下已存在的 ALL.csv（优先）或各类型 csv，
    建立去重键集合：
      (compound_chembl_id, value_type, value_num, value_units, value_relation, pchembl_value)
    """
    keys = set()

    def _accumulate_from_csv(path: str):
        if not os.path.exists(path):
            return
        usecols = [
            "compound_chembl_id", "value_type", "value_num",
            "value_units", "value_relation", "pchembl_value"
        ]
        try:
            for chunk in pd.read_csv(path, usecols=usecols, chunksize=200000):
                chunk = chunk.fillna({"value_units": "", "value_relation": "", "pchembl_value": ""})
                for row in chunk.itertuples(index=False):
                    keys.add((
                        row.compound_chembl_id, row.value_type, row.value_num,
                        row.value_units, row.value_relation, row.pchembl_value
                    ))
        except ValueError:
            # 旧文件列名不完整：全读+补列
            df = pd.read_csv(path)
            for col in usecols:
                if col not in df.columns:
                    df[col] = "" if col in ("value_units", "value_relation", "pchembl_value") else None
            df = df[usecols]
            for row in df.itertuples(index=False):
                keys.add((
                    row.compound_chembl_id, row.value_type, row.value_num,
                    row.value_units, row.value_relation, row.pchembl_value
                ))

    all_csv = os.path.join(target_dir, "ALL.csv")
    if os.path.exists(all_csv):
        _accumulate_from_csv(all_csv)
        return keys

    for p in glob.glob(os.path.join(target_dir, "*.csv")):
        base = os.path.basename(p).lower()
        if base == "all.csv":
            continue
        _accumulate_from_csv(p)
    return keys

def append_rows_to_csv(target_dir: str, rows: List[dict]):
    """
    将一批 rows 按 value_type 分文件写；ALL.csv 同步追加。
    文件不存在时写 header，存在时追加不含 header。
    """
    if not rows:
        return
    df_all = pd.DataFrame(rows)

    # 按类型分别写
    for stype, grp in df_all.groupby("value_type"):
        fpath = os.path.join(target_dir, f"{safe_filename(stype)}.csv")
        write_header = not os.path.exists(fpath)
        grp.to_csv(fpath, index=False, header=write_header, mode="a", encoding="utf-8")

    # ALL.csv
    f_all = os.path.join(target_dir, "ALL.csv")
    write_header_all = not os.path.exists(f_all)
    df_all.to_csv(f_all, index=False, header=write_header_all, mode="a", encoding="utf-8")

def adopt_done_from_disk(state: dict):
    """
    启动时扫描 BASE_DIR 下带 _complete.flag 的目录，
    自动纳入 done_targets（便于采纳以前完整跑过的靶点）。
    """
    if not os.path.isdir(BASE_DIR):
        return
    done = set(state.get("done_targets", []))
    for name in os.listdir(BASE_DIR):
        path = os.path.join(BASE_DIR, name)
        if not os.path.isdir(path):
            continue
        if "__" not in name:
            continue  # 只接受 <TargetName>__<CHEMBL_ID> 格式
        flag = os.path.join(path, "_complete.flag")
        if os.path.exists(flag):
            try:
                chembl_id = name.split("__", 1)[1]
                done.add(chembl_id)
            except Exception:
                pass
    state["done_targets"] = sorted(done)
    save_state(state)

# ========== ChEMBL 抓取 ==========
def fetch_all_targets(limit=5000):
    print(f"[INFO] Fetching up to {limit} targets...", flush=True)
    qs = new_client.target.all().only(["target_chembl_id", "pref_name"])
    targets = list(qs[:limit])
    print(f"[INFO] Got {len(targets)} targets.", flush=True)
    return targets

def activities_generator(target_id: str,
                         start_offset: int = 0,
                         batch_size: int = BATCH_SIZE,
                         max_records: int = MAX_RECORDS_PER_TARGET) -> Iterable[Tuple[int, List[dict]]]:
    """
    逐批产生活性记录：(offset, batch_list)
    """
    print(f"   [STEP] Streaming activities for {target_id} from offset {start_offset} ...", flush=True)
    offset = start_offset
    total = 0
    while True:
        act_q = new_client.activity.filter(
            target_chembl_id=target_id,
            standard_type__in=list(WHITE_LIST)
        ).only([
            "molecule_chembl_id",
            "standard_type",
            "standard_value",
            "standard_units",
            "standard_relation",
            "pchembl_value"
        ])
        batch = list(act_q[offset:offset + batch_size])
        if not batch:
            break
        total += len(batch)
        yield offset, batch
        offset += batch_size
        if offset >= max_records:
            print(f"     [WARN] Too many records for {target_id}, stopping at {max_records}.", flush=True)
            break
    print(f"   [OK] Streamed {total} activity rows for {target_id}.", flush=True)

def fetch_smiles_for_compounds(mol_ids: List[str]) -> Dict[str, str]:
    if not mol_ids:
        return {}
    mol_qs = new_client.molecule.filter(molecule_chembl_id__in=list(mol_ids)) \
                                .only(["molecule_chembl_id", "molecule_structures"])
    smiles_map: Dict[str, str] = {}
    n = 0
    for m in mol_qs:
        cid = m.get("molecule_chembl_id")
        structures = m.get("molecule_structures") or {}
        smiles_map[cid] = (structures or {}).get("canonical_smiles")
        n += 1
        if n % 100 == 0:
            print(f"     [DEBUG] SMILES fetched: {n}", flush=True)
    return smiles_map

# ========== 处理单靶点 ==========
def process_one_target(state: dict, t_id: str, t_name: str):
    target_dir = os.path.join(BASE_DIR, f"{safe_filename(t_name)}__{safe_filename(t_id)}")
    ensure_dir(target_dir)

    # 加载历史已存在键（ALL.csv 或各类型 csv），便于增量去重
    print(f"   [INIT] Loading existing index for {t_id} ...", flush=True)
    existing_keys = build_existing_keys(target_dir)
    print(f"   [INIT] Existing keys: {len(existing_keys)}", flush=True)

    # 从 state 中恢复 offset（若 current 指向该靶点）
    start_offset = 0
    if state.get("current") and state["current"].get("target_id") == t_id:
        start_offset = int(state["current"].get("offset", 0))

    set_current_target(state, t_id, start_offset)

    batch_count = 0
    appended_total = 0

    for offset, batch in activities_generator(t_id, start_offset=start_offset):
        # 更新 offset（即使随后崩溃，也能从 next_offset 继续）
        next_offset = offset + BATCH_SIZE
        set_current_target(state, t_id, next_offset)

        # 提取本批次分子并抓 SMILES
        mol_ids = {r.get("molecule_chembl_id") for r in batch if r.get("molecule_chembl_id")}
        smiles_map = fetch_smiles_for_compounds(list(mol_ids))

        # 转行并做历史去重
        new_rows: List[dict] = []
        for a in batch:
            stype = (a.get("standard_type") or "").strip()
            if stype not in WHITE_LIST:
                continue
            value = a.get("standard_value")
            if value is None:
                continue
            mol_id = a.get("molecule_chembl_id")
            key = (
                mol_id,
                stype,
                value,
                a.get("standard_units") or "",
                a.get("standard_relation") or "",
                a.get("pchembl_value") or ""
            )
            if key in existing_keys:
                continue  # 已存在：跳过
            existing_keys.add(key)

            new_rows.append({
                "target_name": t_name,
                "target_id": t_id,
                "compound_chembl_id": mol_id,
                "compound_smiles": smiles_map.get(mol_id, ""),
                "value_num": value,
                "value_units": key[3],
                "value_relation": key[4],
                "pchembl_value": key[5],
                "value_type": stype
            })

        if new_rows:
            append_rows_to_csv(target_dir, new_rows)
            appended_total += len(new_rows)
            print(f"   [SAVED] {t_id} offset {offset}: +{len(new_rows)} rows (total new {appended_total})", flush=True)
        else:
            print(f"   [SKIP] {t_id} offset {offset}: no new rows", flush=True)

        batch_count += 1
        if CHECKPOINT_EVERY_BATCH > 0 and (batch_count % CHECKPOINT_EVERY_BATCH == 0):
            save_state(state)

    mark_target_done(state, t_id, target_dir)
    print(f"   [DONE] {t_id} fully processed. Newly appended rows: {appended_total}", flush=True)

# ========== 主流程 ==========
def main():
    ensure_dir(BASE_DIR)
    state = load_state()
    adopt_done_from_disk(state)  # 采纳历史完成的目录

    targets = fetch_all_targets(limit=TARGET_LIMIT)

    # 如果 state.current 指向某个靶点，把它排到最前面
    current_first = None
    if state.get("current"):
        cur_id = state["current"].get("target_id")
        if cur_id:
            for t in targets:
                if t["target_chembl_id"] == cur_id:
                    current_first = t
                    break

    done_set = set(state.get("done_targets", []))

    # 遍历顺序：current（若有） -> 未完成 -> 已完成（仅跳过，不再处理）
    ordered: List[dict] = []
    if current_first:
        ordered.append(current_first)
    for t in targets:
        if current_first and t["target_chembl_id"] == current_first["target_chembl_id"]:
            continue
        if t["target_chembl_id"] not in done_set:
            ordered.append(t)
    for t in targets:
        if t["target_chembl_id"] in done_set:
            ordered.append(t)

    for idx, t in enumerate(tqdm(ordered, desc="Processing targets"), start=1):
        t_id = t["target_chembl_id"]
        t_name = t.get("pref_name") or f"Unknown_{t_id}"
        if t_id in done_set:
            # 历史完成的直接跳过
            continue
        print(f"\n[PROCESS] ({idx}/{len(ordered)}) Target: {t_name} ({t_id})", flush=True)
        try:
            process_one_target(state, t_id, t_name)
        except Exception as e:
            print(f"   [ERROR] {t_id}: {e}", flush=True)
            save_state(state)
            #不中断全局循环，继续后续；若想严格终止可改为 raise
            continue

if __name__ == "__main__":
    main()
