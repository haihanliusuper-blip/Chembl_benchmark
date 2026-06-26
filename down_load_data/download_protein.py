# -*- coding: utf-8 -*-
# pip install chembl_webresource_client requests pandas tqdm

import os
import re
import time
import random
import hashlib
import json
import requests
import pandas as pd
from tqdm import tqdm
from chembl_webresource_client.new_client import new_client

# ================== 可调配置 ==================
BASE_DIR = "ChEMBL_Targets_MIN"
TARGET_LIMIT = 6000           # 当跑全库时，这是全库总上限
ONLY_CLASSES = None           # 设为 ["Hydrolase", "Kinase"] 按类别；设为 None 跑全库
PREFER_TAXID = 9606           # 只要人源（用于 UniProt 检索过滤）
CANDIDATE_SIZE = 15           # 每个名字最多取多少 UniProt 候选
HTTP_TIMEOUT = 25
RETRY = 3
BACKOFF_BASE = 0.6
BACKOFF_JITTER = 0.3

# ===== 断点续跑与覆盖策略（环境变量可改） =====
RESUME = bool(int(os.getenv("RESUME", "1")))        # 1=默认续跑
OVERWRITE = bool(int(os.getenv("OVERWRITE", "0")))  # 1=强制重跑覆盖
SKIP_FAILED = bool(int(os.getenv("SKIP_FAILED", "0")))  # 1=对上次失败的目标直接跳过

# =============== 日志 ===============
def ts(): return time.strftime("%H:%M:%S")
def step(msg):  print(f"[{ts()}][STEP] {msg}")
def warn(msg):  print(f"[{ts()}][WARN] {msg}")
def dbg(msg):   print(f"[{ts()}][DBG ] {msg}")

# =============== 工具 ===============
def safe_filename(s: str) -> str:
    return re.sub(r'[^a-zA-Z0-9._-]', "_", str(s)).strip("_")[:120] or "Unknown"

def mk_session():
    s = requests.Session()
    s.headers.update({"User-Agent": "name2uniprot-human-min/0.3"})
    return s

def _sleep_backoff(i: int):
    time.sleep((BACKOFF_BASE ** i) + random.uniform(0, BACKOFF_JITTER))

def http_get(s, url, params=None, tag=""):
    last = None
    for i in range(1, RETRY+1):
        try:
            r = s.get(url, params=params, timeout=HTTP_TIMEOUT)
            if r.status_code == 200:
                return r
            if r.status_code in (429, 500, 502, 503, 504):
                last = f"{r.status_code}"
                _sleep_backoff(i)
            else:
                last = f"{r.status_code} {r.text[:200]}"
                break
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
            _sleep_backoff(i)
    warn(f"GET failed {tag}: {last}")
    return None

def http_get_json(s, url, params=None, tag=""):
    r = http_get(s, url, params=params, tag=tag)
    if not r: return None
    try:
        return r.json()
    except Exception as e:
        warn(f"JSON parse failed {tag}: {e}")
        return None

def sha1_seq_from_fasta(txt: str) -> str:
    if not txt: return ""
    body = re.sub(r"^>.*\n", "", txt, flags=re.M)
    body = re.sub(r"\s+", "", body)
    return hashlib.sha1(body.encode("utf-8")).hexdigest()[:12] if body else ""

# ====== 原子写 & 断点标记 工具 ======
def atomic_write_text(path: str, text: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)

def atomic_write_df(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

def done_mark_path(tdir: str) -> str:
    return os.path.join(tdir, "DONE.ok")

def fail_mark_path(tdir: str) -> str:
    return os.path.join(tdir, "FAILED.err")

def is_done(tdir: str) -> bool:
    p = done_mark_path(tdir)
    seq = os.path.join(tdir, "SEQUENCE", "sequence_best.fasta")
    return os.path.exists(p) and os.path.exists(seq)

def is_failed(tdir: str) -> bool:
    return os.path.exists(fail_mark_path(tdir))

def mark_done(tdir: str, note: str = ""):
    atomic_write_text(done_mark_path(tdir), f"{ts()} DONE\n{note}\n")

def mark_failed(tdir: str, note: str = ""):
    atomic_write_text(fail_mark_path(tdir), f"{ts()} FAILED\n{note}\n")

def clear_marks(tdir: str):
    for p in [done_mark_path(tdir), fail_mark_path(tdir)]:
        try: os.remove(p)
        except FileNotFoundError: pass

# =============== ChEMBL 取 targets（只要名字） ===============
def chembl_targets_by_class(class_name: str, limit: int):
    qs = new_client.target.only(["target_chembl_id", "pref_name"])
    out, seen = [], set()
    try:
        for r in qs.filter(protein_classifications__l2__icontains=class_name)[:limit]:
            tid = r.get("target_chembl_id")
            if tid and tid not in seen:
                out.append(r); seen.add(tid)
    except: pass
    if len(out) < limit and class_name.lower() == "kinase":
        try:
            for r in qs.filter(protein_classifications__l3__icontains="Protein Kinase")[:limit]:
                tid = r.get("target_chembl_id")
                if tid and tid not in seen:
                    out.append(r); seen.add(tid)
        except: pass
    if len(out) < limit:
        try:
            for r in qs.filter(pref_name__icontains=class_name.lower())[:limit]:
                tid = r.get("target_chembl_id")
                if tid and tid not in seen:
                    out.append(r); seen.add(tid)
        except: pass
    return out[:limit]

def chembl_targets_all(limit: int):
    qs = new_client.target.only(["target_chembl_id", "pref_name"])
    out, seen = [], set()
    try:
        for r in qs[:limit]:
            tid = r.get("target_chembl_id")
            if tid and tid not in seen:
                out.append(r); seen.add(tid)
            if len(out) >= limit:
                break
    except Exception as e:
        warn(f"Fetch all targets failed: {e}")
    return out

# =============== 读取单个 ChEMBL target 的完整元数据 ===============
def chembl_target_meta(tid: str) -> dict:
    """
    返回:
    {
      'target_id', 'target_name', 'target_type', 'organism',
      'components': [{'component_type','accession','organism','tax_id'} ...]
    }
    """
    try:
        recs = list(new_client.target.filter(target_chembl_id=tid))
        if not recs:
            return {"target_id": tid, "target_name": "", "target_type": "", "organism": "", "components": []}
        r = recs[0]
        comps = []
        for c in (r.get("target_components") or []):
            comp = c.get("component") or {}
            comps.append({
                "component_type": comp.get("component_type") or c.get("component_type") or "",
                "accession": comp.get("accession") or "",
                "organism": comp.get("organism") or "",
                "tax_id": comp.get("tax_id") or ""
            })
        return {
            "target_id": r.get("target_chembl_id") or tid,
            "target_name": r.get("pref_name") or "",
            "target_type": r.get("target_type") or "",
            "organism": r.get("organism") or "",
            "components": comps
        }
    except Exception as e:
        warn(f"chembl_target_meta({tid}) failed: {e}")
        return {"target_id": tid, "target_name": "", "target_type": "", "organism": "", "components": []}

def coarse_kind(target_type: str) -> str:
    tt = (target_type or "").upper()
    if "CELL" in tt:
        return "Cell-line"
    if "PROTEIN" in tt:
        return "Protein"
    return "Others"

# =============== UniProt 名称检索（合法字段） ===============
UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_ENTRY  = "https://rest.uniprot.org/uniprotkb/{acc}.json"
UNIPROT_FASTA  = "https://rest.uniprot.org/uniprotkb/{acc}.fasta"

GREEK_MAP = {
    "α":"alpha","β":"beta","γ":"gamma","δ":"delta","ε":"epsilon","ζ":"zeta","η":"eta","θ":"theta","ι":"iota",
    "κ":"kappa","λ":"lambda","μ":"mu","ν":"nu","ξ":"xi","ο":"omicron","π":"pi","ρ":"rho","σ":"sigma","ς":"sigma",
    "τ":"tau","υ":"upsilon","φ":"phi","χ":"chi","ψ":"psi","ω":"omega",
    "Α":"alpha","Β":"beta","Γ":"gamma","Δ":"delta","Ε":"epsilon","Ζ":"zeta","Η":"eta","Θ":"theta","Ι":"iota","Κ":"kappa",
    "Λ":"lambda","Μ":"mu","Ν":"nu","Ξ":"xi","Ο":"omicron","Π":"pi","Ρ":"rho","Σ":"sigma","Τ":"tau","Υ":"upsilon","Φ":"phi","Χ":"chi","Ψ":"psi","Ω":"omega"
}
def norm_greek(s: str) -> str:
    return "".join(GREEK_MAP.get(ch, ch) for ch in (s or ""))

def build_name_queries(name: str, taxid: int = PREFER_TAXID):
    n0 = (name or "").strip()
    n1 = norm_greek(n0)
    vs = sorted(set([n0, n1, n1.replace("-", " "), n1.upper(), n1.lower()]))

    base = []
    for v in vs:
        if re.fullmatch(r"[A-Za-z0-9\-]+", v):
            base.append(f"(gene:{v})")
    for v in vs:
        if len(v) >= 3:
            base.append(f"(\"{v}\")")
    for v in vs:
        base.append(f"({v})")

    seen, ordered = set(), []
    for q in base:
        if q not in seen:
            seen.add(q); ordered.append(q)

    def add_filters(q):
        return f"({q}) AND reviewed:true AND organism_id:{taxid}"
    return [add_filters(q) for q in ordered]

def uniprot_search_accessions(s, query: str, size: int):
    params = {
        "query": query,
        "format": "json",
        "size": str(size),
        "fields": "accession,id,reviewed,organism_id,length,protein_name,gene_names"
    }
    js = http_get_json(s, UNIPROT_SEARCH, params=params, tag="search")
    if not js: return []
    accs = []
    for it in js.get("results", []):
        acc = it.get("primaryAccession") or it.get("accession") or it.get("uniProtkbId")
        if acc:
            accs.append(acc)
    return accs

def search_accessions_by_name(s, name: str, size: int = CANDIDATE_SIZE):
    accs, seen = [], set()
    for q in build_name_queries(name, taxid=PREFER_TAXID):
        xs = uniprot_search_accessions(s, q, size=size)
        for a in xs:
            if a not in seen:
                seen.add(a); accs.append(a)
        if len(accs) >= size:
            break
    dbg(f"[name→UniProt] '{name}' -> {len(accs)} candidates")
    return accs[:size]

def fetch_entry(s, acc: str):
    return http_get_json(s, UNIPROT_ENTRY.format(acc=acc), tag=f"entry {acc}") or {}

def fetch_fasta(s, acc: str) -> str:
    r = http_get(s, UNIPROT_FASTA.format(acc=acc), tag=f"fasta {acc}")
    if not r: return ""
    try:
        return r.text if hasattr(r, "text") else r.content.decode("utf-8", errors="ignore")
    except Exception:
        try:
            return r.content.decode("utf-8", errors="ignore")
        except Exception:
            return ""

def pick_best_human_reviewed(s, accs):
    best = None
    for acc in accs:
        js = fetch_entry(s, acc)
        org = (js.get("organism") or {}).get("taxonId")
        et  = (js.get("entryType") or "").upper()  # 'SWISS-PROT' => reviewed
        reviewed = (et == "SWISS-PROT")
        if org == PREFER_TAXID and reviewed:
            best = acc; break
        if best is None:
            best = acc
    return best

# =============== 保存序列（优先用 ChEMBL component 的 accession） ===============
def save_best_sequence_for_target(s, out_dir: str, tmeta: dict):
    """
    tmeta: chembl_target_meta() 的返回
    先看 components 里是否有 PROTEIN accession（人源优先），有则直接用；否则回退到名字检索。
    返回: {'best_accession': str, 'sequence_sha1': str, 'success': bool}
    """
    os.makedirs(out_dir, exist_ok=True)
    seq_dir = os.path.join(out_dir, "SEQUENCE")
    os.makedirs(seq_dir, exist_ok=True)

    # 把 components 快照保存，便于定位为何匹配/不匹配
    components_path = os.path.join(out_dir, "components.json")
    try:
        atomic_write_text(components_path, json.dumps(tmeta.get("components", []), ensure_ascii=False, indent=2))
    except Exception as e:
        warn(f"write components.json failed: {e}")

    # 1) 从 components 提取候选 accession
    accessions = []
    for c in (tmeta.get("components") or []):
        if str(c.get("component_type","")).upper() == "PROTEIN" and c.get("accession"):
            accessions.append((c.get("accession"), c.get("tax_id") or "", c.get("organism") or ""))

    # 优先人源
    accs_priority = [a for a,tax,org in accessions if str(tax)==str(PREFER_TAXID) or "Homo sapiens" in str(org)]
    if not accs_priority:
        accs_priority = [a for a,_,_ in accessions]

    # 2) 若 component 没给 accession，再走名字检索
    if not accs_priority:
        name = tmeta.get("target_name") or ""
        accs_priority = search_accessions_by_name(s, name, size=CANDIDATE_SIZE)

    # 保存候选索引
    idx_rows = []
    for acc in (accs_priority if isinstance(accs_priority, list) else []):
        if isinstance(acc, tuple):
            a = acc[0]
        else:
            a = acc
        js = fetch_entry(s, a)
        org = (js.get("organism") or {}).get("taxonId")
        et  = (js.get("entryType") or "").upper()
        length = (js.get("sequence") or {}).get("length", "")
        idx_rows.append({
            "query_name": tmeta.get("target_name",""),
            "accession": a,
            "organism_id": org,
            "reviewed": (et == "SWISS-PROT"),
            "length": length
        })
    seq_index_csv = os.path.join(seq_dir, "seq_index.csv")
    atomic_write_df(pd.DataFrame(idx_rows if idx_rows else [{
        "query_name": tmeta.get("target_name",""), "accession":"", "organism_id":"", "reviewed":"", "length":""
    }]), seq_index_csv)
    step(f"SAVED {seq_index_csv} ({len(idx_rows) if idx_rows else 1} rows)")

    if not accs_priority:
        warn(f"No UniProt candidates for '{tmeta.get('target_name','')}'")
        return {"best_accession":"", "sequence_sha1":"", "success": False}

    # 3) 选最佳（reviewed 且人源优先）
    accs_plain = [a if isinstance(a, str) else a[0] for a in accs_priority]
    best = pick_best_human_reviewed(s, accs_plain)
    if not best:
        warn("No best accession chosen")
        return {"best_accession":"", "sequence_sha1":"", "success": False}

    # 4) 拉 FASTA + 落盘
    fasta = fetch_fasta(s, best)
    if fasta.strip():
        sha = sha1_seq_from_fasta(fasta)
        fpath = os.path.join(seq_dir, f"sequence_{safe_filename(best)}.fasta")
        atomic_write_text(fpath, fasta)
        step(f"SAVED {fpath} (sha1={sha})")
        alias = os.path.join(seq_dir, "sequence_best.fasta")
        if alias != fpath:
            atomic_write_text(alias, fasta)
        return {"best_accession": best, "sequence_sha1": sha, "success": True}
    else:
        warn(f"FASTA fetch failed for {best}")
        return {"best_accession":"", "sequence_sha1":"", "success": False}

# =============== 主流程（批量） ===============
def run_batch():
    os.makedirs(BASE_DIR, exist_ok=True)
    s = mk_session()
    global_rows = []

    def handle_one_target(tinfo, class_tag="ALL"):
        tid = tinfo.get("target_chembl_id")
        tname = tinfo.get("pref_name") or f"Unknown_{tid}"
        meta = chembl_target_meta(tid)
        ttype = meta.get("target_type","")
        org   = meta.get("organism","")

        tdir = os.path.join(BASE_DIR if class_tag=="ALL" else os.path.join(BASE_DIR, class_tag),
                            f"{safe_filename(tname)}__{safe_filename(tid)}")
        os.makedirs(tdir, exist_ok=True)

        # 续跑/覆盖策略
        if OVERWRITE:
            clear_marks(tdir)
        else:
            if RESUME and is_done(tdir):
                step(f"[RESUME] skip done: {tdir}")
                idx_csv = os.path.join(tdir, "target_index.csv")
                if os.path.exists(idx_csv):
                    try:
                        mini_old = pd.read_csv(idx_csv).iloc[0].to_dict()
                        return mini_old
                    except Exception:
                        pass
                return {
                    "class": class_tag,
                    "target_id": tid,
                    "target_name": tname,
                    "target_type": meta.get("target_type",""),
                    "organism": meta.get("organism",""),
                    "coarse_kind": coarse_kind(meta.get("target_type","")),
                    "best_accession": "",
                    "sequence_sha1": "",
                    "note": "resume-skip-done"
                }

            if RESUME and SKIP_FAILED and is_failed(tdir):
                step(f"[RESUME] skip failed: {tdir}")
                return {
                    "class": class_tag,
                    "target_id": tid,
                    "target_name": tname,
                    "target_type": meta.get("target_type",""),
                    "organism": meta.get("organism",""),
                    "coarse_kind": coarse_kind(meta.get("target_type","")),
                    "best_accession": "",
                    "sequence_sha1": "",
                    "note": "resume-skip-failed"
                }

        step(f"{tname} ({tid}) | target_type={ttype or 'NA'} | organism={org or 'NA'}")
        info = save_best_sequence_for_target(s, tdir, meta)

        mini = {
            "class": class_tag,
            "target_id": tid,
            "target_name": tname,
            "target_type": ttype,
            "organism": org,
            "coarse_kind":  ttype,  # Protein / Cell-line / Others
            "best_accession": info.get("best_accession",""),
            "sequence_sha1": info.get("sequence_sha1",""),
            "note": "" if info.get("success") else "no uniprot hit"
        }

        # 标记完成/失败 + 保存 per-target 索引
        try:
            if info.get("success"):
                mark_done(tdir, note=f"best={info.get('best_accession','')}")
                try: os.remove(fail_mark_path(tdir))
                except FileNotFoundError: pass
            else:
                mark_failed(tdir, note="no-sequence")
                try: os.remove(done_mark_path(tdir))
                except FileNotFoundError: pass
        except Exception as e:
            warn(f"marking error: {e}")

        atomic_write_df(pd.DataFrame([mini]), os.path.join(tdir, "target_index.csv"))
        return mini

    if ONLY_CLASSES:
        for CLASS in ONLY_CLASSES:
            step(f"==== Class: {CLASS} ====")
            targets = chembl_targets_by_class(CLASS, TARGET_LIMIT)
            step(f"Picked {len(targets)} targets")
            class_dir = os.path.join(BASE_DIR, CLASS)
            os.makedirs(class_dir, exist_ok=True)
            for t in tqdm(targets, desc=f"[{CLASS}]"):
                mini = handle_one_target(t, class_tag=CLASS)
                global_rows.append(mini)
    else:
        CLASS = "ALL"
        step(f"==== All Targets (limit={TARGET_LIMIT}) ====")
        targets = chembl_targets_all(TARGET_LIMIT)
        step(f"Picked {len(targets)} targets")
        for t in tqdm(targets, desc=f"[{CLASS}]"):
            mini = handle_one_target(t, class_tag=CLASS)
            global_rows.append(mini)

    # 全局索引（原子写）
    gpath = os.path.join(BASE_DIR, "GLOBAL_index.csv")
    atomic_write_df(pd.DataFrame(global_rows), gpath)
    step(f"GLOBAL index saved to {gpath}")

    # ====== 概览统计 ======
    df_global = pd.DataFrame(global_rows)
    if not df_global.empty:
        step("==== SUMMARY ====")
        by_type = df_global["target_type"].fillna("NA").value_counts().to_dict()
        by_kind = df_global["coarse_kind"].fillna("NA").value_counts().to_dict()
        print("[BY target_type]")
        for k,v in by_type.items():
            print(f"  - {k}: {v}")
        print("[BY coarse_kind]")
        for k,v in by_kind.items():
            print(f"  - {k}: {v}")
    print("[DONE]")

# =============== 单名字快速测试 ===============
def run_single_test(test_name="RARγ", chembl_id=None):
    s = mk_session()
    out_dir = os.path.join(BASE_DIR, f"TEST_{safe_filename(test_name)}")
    if chembl_id:
        meta = chembl_target_meta(chembl_id)
        info = save_best_sequence_for_target(s, out_dir, meta)
        print("TEST best:", info)
    else:
        meta = {"target_name": test_name, "components": []}
        info = save_best_sequence_for_target(s, out_dir, meta)
        print("TEST best:", info)

if __name__ == "__main__":
    # 默认跑批量；想测试可以改用 run_single_test()
    run_batch()
    # run_single_test(chembl_id="CHEMBL2842")  # 例：已知ID
    # run_single_test("RARγ")                  # 例：按名字
