#!/usr/bin/env python3
"""Extend the verified set with SWE-Live instances so the batch math is healthy.

swerebench_verified has only 22 train instances (the verified∩SWE-rebench ceiling) —
too few for a full batch of 32 and thin for batch 16. This builds
`swerebench_verified_live` = the 22 verified-train + 4 verified-val rows (reused as-is)
PLUS SWE-Live∩SWE-rebench instances (repos from live_swerebench_overlap.tsv) to reach
64 train / 8 val. Same verl format + make_test_spec + swerebench/ images.

Branch test-refinement-test-to-openai.
"""
import os, json, dataclasses
import pandas as pd
from datasets import load_dataset
from collections import defaultdict, Counter
from swebench_fork_swerebench.harness.test_spec.test_spec import make_test_spec as mts_swe

TRAIN_TARGET, VAL_TARGET = 64, 8
SRC_VERIFIED = "data_examples/rl_data/swerebench_verified"
OUT = "data_examples/rl_data/swerebench_verified_live"

VERIFIED = {"astropy/astropy","matplotlib/matplotlib","scikit-learn/scikit-learn",
            "mwaskom/seaborn","sphinx-doc/sphinx","sympy/sympy","pydata/xarray"}

def _overlap_repos(tsv, col="swe_rebench_repo"):
    rows=[l.rstrip("\n").split("\t") for l in open(tsv) if not l.startswith("#") and l.strip()]
    hdr=rows[0]; idx=hdr.index(col)
    # strip trailing "  # CHECK..." annotations some rows carry
    return {r[idx].split("#")[0].strip() for r in rows[1:] if len(r)>idx}

LIVE = _overlap_repos("live_swerebench_overlap.tsv") - VERIFIED
print(f"live∩swerebench repos: {len(LIVE)}")

# --- reuse the already-built + validated verified rows ---
vtrain = pd.read_parquet(f"{SRC_VERIFIED}/train_00000_verl.parquet")
vval   = pd.read_parquet(f"{SRC_VERIFIED}/test_00000_verl.parquet")
print(f"verified rows reused: train={len(vtrain)} val={len(vval)}")
COLS = list(vtrain.columns)

# --- template (system + user wrapper) + schema from the verified/comprehensive rows ---
trow = vtrain.iloc[0]
SYS_MSG = trow["prompt"][0]["content"]
USER_TMPL = trow["prompt"][1]["content"]
TMPL_PS = json.loads(trow["extra_info"])["problem_statement"]
assert TMPL_PS in USER_TMPL
PREFIX, SUFFIX = USER_TMPL.split(TMPL_PS, 1)
LOCAL_KEYS = list(json.loads(trow["extra_info"]).keys())

def to_verl_row(inst):
    inst = dict(inst)
    try:
        ts = mts_swe(inst)
        mts = json.dumps(dataclasses.asdict(ts) if dataclasses.is_dataclass(ts) else dict(ts.__dict__))
    except Exception:
        return None
    ei = {}
    for k in LOCAL_KEYS:
        ei[k] = inst.get("docker_image") if k=="image_name" else (mts if k=="make_test_spec" else inst.get(k))
    user = PREFIX + (inst.get("problem_statement") or "") + SUFFIX
    return {"data_source":"swe",
            "prompt":[{"role":"system","content":SYS_MSG},{"role":"user","content":user}],
            "ability":"swe","reward_model":{"style":"rule","ground_truth":""},
            "extra_info":json.dumps(ei)}

print("loading nebius/SWE-rebench ...")
d = load_dataset("nebius/SWE-rebench", split="test")
cand=[x for x in d if x.get("repo") in LIVE and x.get("FAIL_TO_PASS") and x.get("patch")
      and x.get("test_patch") and x.get("docker_image")]
by_repo=defaultdict(list)
for x in sorted(cand, key=lambda r:r["instance_id"]):
    by_repo[x["repo"]].append(x)
repos=sorted(by_repo)
print(f"live candidates: {len(cand)} across {len(repos)} repos")

need_train = TRAIN_TARGET - len(vtrain)
need_val   = VAL_TARGET   - len(vval)
print(f"need live: train={need_train} val={need_val}")

# round-robin across live repos for diversity; build make_test_spec; keep valid
live_rows=[]; live_repos=[]
order=[x for r in repos for x in []]  # placeholder
picked=[]
while True:
    progressed=False
    for r in repos:
        if by_repo[r]:
            picked.append(by_repo[r].pop(0)); progressed=True
    if not progressed: break
for inst in picked:
    if len(live_rows) >= need_train + need_val: break
    row=to_verl_row(inst)
    if row is not None:
        live_rows.append(row); live_repos.append(inst["repo"])
print(f"valid live rows: {len(live_rows)}")
assert len(live_rows) >= need_train + need_val, f"only {len(live_rows)} live valid"

# split live: first need_val (stratified across distinct repos) -> val, rest -> train
val_idx, seen = [], set()
for i, rp in enumerate(live_repos):
    if rp not in seen:
        seen.add(rp); val_idx.append(i)
    if len(val_idx) >= need_val: break
vset=set(val_idx)
live_val=[live_rows[i] for i in val_idx]
live_train=[r for i,r in enumerate(live_rows) if i not in vset][:need_train]

train_df = pd.concat([vtrain, pd.DataFrame(live_train)[COLS]], ignore_index=True)
val_df   = pd.concat([vval,   pd.DataFrame(live_val)[COLS]],   ignore_index=True)
os.makedirs(OUT, exist_ok=True)
train_df.to_parquet(f"{OUT}/train_00000_verl.parquet")
val_df.to_parquet(f"{OUT}/test_00000_verl.parquet")
print(f"WROTE {OUT}: train={len(train_df)} val={len(val_df)}")

# sanity
allrepos=[json.loads(x)["repo"] for x in pd.concat([train_df,val_df])["extra_info"]]
nver=sum(1 for r in allrepos if r in VERIFIED)
print("columns OK:", list(train_df.columns)==COLS)
print(f"verified instances: {nver} | live instances: {len(allrepos)-nver}")
print("repo balance (top):", Counter(allrepos).most_common(12))
