#!/usr/bin/env python3
"""Build a 32/4 training set whose instances come from SWE-rebench (nebius/SWE-rebench)
but whose REPOS are all SWE-bench Verified — the 7 repos in the
SWE-bench-Verified ∩ SWE-rebench overlap (verified_swerebench_overlap.tsv):
  astropy, matplotlib, scikit-learn, seaborn(mwaskom), sphinx, sympy, xarray(pydata).

Same verl parquet format + prompt template as the original ATS-RL set
(swerebench_comprehensive), with make_test_spec rebuilt via swebench_fork_swerebench and
swerebench/ docker images (pullable at runtime). Output:
  data_examples/rl_data/swerebench_verified/{train,test}_00000_verl.parquet  (32 / 4).

Branch test-refinement-test-to-openai.
"""
import os, json, dataclasses
import pandas as pd
from datasets import load_dataset
from collections import defaultdict
from swebench_fork_swerebench.harness.test_spec.test_spec import make_test_spec as mts_swe

# Cap is dictated by availability: only ~26 verified∩SWE-rebench instances exist
# (scikit-learn & seaborn have ZERO instances in SWE-rebench). We use all of them,
# holding out N_VAL stratified across repos; the rest are train.
N_VAL = 4
TOTAL = 999  # take everything available; split below

# --- the 7 SWE-bench-Verified ∩ SWE-rebench repos (from the TSV) ---
def _read_overlap_repos(tsv="verified_swerebench_overlap.tsv"):
    rows = [l.rstrip("\n").split("\t") for l in open(tsv) if not l.startswith("#") and l.strip()]
    hdr, data = rows[0], rows[1:]
    idx = hdr.index("swe_rebench_repo")
    return {r[idx] for r in data}

VERIFIED = _read_overlap_repos()
print("verified∩swerebench repos:", sorted(VERIFIED))

# --- prompt template + extra_info schema from the original ATS-RL set ---
tdf = pd.read_parquet("data_examples/rl_data/swerebench_comprehensive/train_00000_verl.parquet")
trow = tdf.iloc[0]
SYS_MSG = trow["prompt"][0]["content"]
USER_TMPL = trow["prompt"][1]["content"]
TMPL_PS = json.loads(trow["extra_info"])["problem_statement"]
assert TMPL_PS in USER_TMPL, "template problem_statement not found in user message"
PREFIX, SUFFIX = USER_TMPL.split(TMPL_PS, 1)
LOCAL_KEYS = list(json.loads(trow["extra_info"]).keys())

print("loading nebius/SWE-rebench ...")
d = load_dataset("nebius/SWE-rebench", split="test")

# candidates: repo in the verified overlap, complete (F2P + patch + test_patch + image)
cand = []
for x in d:
    if x.get("repo") in VERIFIED \
       and x.get("FAIL_TO_PASS") and x.get("patch") and x.get("test_patch") and x.get("docker_image"):
        cand.append(x)
print(f"candidates (verified-repo, complete): {len(cand)}")

by_repo = defaultdict(list)
for x in sorted(cand, key=lambda r: r["instance_id"]):
    by_repo[x["repo"]].append(x)
repos_ordered = sorted(by_repo.keys())
print("per-repo candidate counts:", {r: len(by_repo[r]) for r in repos_ordered})

# round-robin across the verified repos for diversity
picked = []
while len(picked) < TOTAL * 2:  # over-pick; make_test_spec filter prunes below
    progressed = False
    for r in repos_ordered:
        if by_repo[r]:
            picked.append(by_repo[r].pop(0)); progressed = True
    if not progressed:
        break
print(f"pre-selected {len(picked)} instances across {len(set(p['repo'] for p in picked))} repos")


def to_verl_row(inst):
    """verl row; None if make_test_spec can't be built for this instance."""
    inst = dict(inst)
    try:
        ts = mts_swe(inst)
        mts = json.dumps(dataclasses.asdict(ts) if dataclasses.is_dataclass(ts) else dict(ts.__dict__))
    except Exception:
        return None
    ei = {}
    for k in LOCAL_KEYS:
        if k == "image_name":
            ei[k] = inst.get("docker_image")
        elif k == "make_test_spec":
            ei[k] = mts
        else:
            ei[k] = inst.get(k, None)
    user = PREFIX + (inst.get("problem_statement") or "") + SUFFIX
    return {
        "data_source": "swe",
        "prompt": [{"role": "system", "content": SYS_MSG}, {"role": "user", "content": user}],
        "ability": "swe",
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": json.dumps(ei),
    }


# keep make_test_spec-valid rows, round-robin order, until TOTAL; track repo balance
valid_rows, valid_repos = [], []
leftovers = [x for r in repos_ordered for x in by_repo[r]]
for inst in picked + leftovers:
    if len(valid_rows) >= TOTAL:
        break
    row = to_verl_row(inst)
    if row is not None:
        valid_rows.append(row)
        valid_repos.append(inst["repo"])
print(f"valid rows (make_test_spec built): {len(valid_rows)}")
assert len(valid_rows) > N_VAL, f"only {len(valid_rows)} valid; need > {N_VAL}"

# Hold out N_VAL stratified across DISTINCT repos (first occurrence of each repo),
# everything else is train.
val_idx, seen = [], set()
for i, rp in enumerate(valid_repos):
    if rp not in seen:
        seen.add(rp); val_idx.append(i)
    if len(val_idx) >= N_VAL:
        break
val_set = set(val_idx)
val_rows = [valid_rows[i] for i in val_idx]
train_rows = [r for i, r in enumerate(valid_rows) if i not in val_set]
out = "data_examples/rl_data/swerebench_verified"
os.makedirs(out, exist_ok=True)
pd.DataFrame(train_rows).to_parquet(f"{out}/train_00000_verl.parquet")
pd.DataFrame(val_rows).to_parquet(f"{out}/test_00000_verl.parquet")
print(f"WROTE {out}: train={len(train_rows)} val={len(val_rows)}")

# sanity: schema parity + ALL repos verified
nt = pd.read_parquet(f"{out}/train_00000_verl.parquet")
nv = pd.read_parquet(f"{out}/test_00000_verl.parquet")
assert list(nt.columns) == list(tdf.columns), f"column mismatch {list(nt.columns)} vs {list(tdf.columns)}"
all_repos = [json.loads(x)["repo"] for x in pd.concat([nt, nv])["extra_info"]]
bad = [r for r in all_repos if r not in VERIFIED]
from collections import Counter
print("columns OK |", "repo balance:", dict(Counter(all_repos)))
print(f"ALL repos verified: {len(bad)==0} (non-verified: {bad})")
print("sample train ids:", [json.loads(x)["instance_id"] for x in nt["extra_info"].head(5)])
