#!/usr/bin/env python3
"""Comprehensive SWE-Master RL training set from nebius/SWE-rebench (the public
mirror of the SWE-Master training distribution; docker_image is already the
public Docker Hub `swerebench/...`). Selects N tasks whose images actually
exist on Docker Hub, embeds make_test_spec, writes train/val verl parquets in
the original scripts/data/swe_dataset.py schema.
"""
import json, os, subprocess, sys, random
import pandas as pd
sys.path.insert(0, "/home/temp/swemaster_runpod/DeepSWE_RL/rllm")
sys.path.insert(0, "/home/temp/swemaster_runpod/R2E-Gym/src")
from datasets import load_dataset
from rllm.agents.system_prompts import SWE_SYSTEM_PROMPT, SWE_USER_PROMPT

N_TRAIN = int(os.environ.get("N_TRAIN", "20"))
N_VAL = int(os.environ.get("N_VAL", "4"))
MAX_PROBE = int(os.environ.get("MAX_PROBE", "120"))
# Minimum FAIL_TO_PASS size to keep — useful for the active-test-selection
# reward studies, where small FAIL_TO_PASS degenerates the partial-credit
# signal to near-binary. Default 0 keeps the legacy behavior.
MIN_FTP = int(os.environ.get("MIN_FTP", "0"))
OUT = os.environ.get(
    "OUT_DIR",
    "/home/temp/swemaster_runpod/data_examples/rl_data/swerebench_comprehensive",
)
os.makedirs(OUT, exist_ok=True)

# CACHED_ONLY=1 → select only tasks whose image is ALREADY pulled locally
# (uses `docker image inspect`, a local call). This guarantees zero Docker Hub
# pulls at training time → no rate-limit NO_CONTAINER failures. Default 0
# keeps the remote `docker manifest inspect` existence check.
CACHED_ONLY = os.environ.get("CACHED_ONLY", "0").strip().lower() in ("1", "true", "yes")

def image_exists(img):
    if CACHED_ONLY:
        # local-only check: image must already be in the docker image store
        return subprocess.run(["docker", "image", "inspect", img],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    # cheap remote existence check, no full pull
    return subprocess.run(["docker", "manifest", "inspect", img],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0

print("loading nebius/SWE-rebench (test split)...")
ds = load_dataset("nebius/SWE-rebench", split="test")
from swebench_fork_swerebench.harness.test_spec.test_spec import make_test_spec as mts_swerebench

# Keep only rows that already point at a public swerebench/ Docker Hub image,
# then prefer shorter problem statements (more tractable for a 4B agent).
idx = [i for i in range(len(ds)) if "swerebench/" in (ds[i].get("docker_image") or "")]
idx.sort(key=lambda i: len(ds[i].get("problem_statement") or "") or 10**9)
print(f"{len(idx)} rows have a public swerebench/ image; probing up to {MAX_PROBE}")

rows, picked = [], []
need = N_TRAIN + N_VAL
for c, i in enumerate(idx[:MAX_PROBE]):
    if len(rows) >= need:
        break
    r = dict(ds[i])
    img = r.get("docker_image") or r.get("image_name")
    if ":" not in img.split("/")[-1]:
        img = img + ":latest"
    if not image_exists(img):
        continue
    # Optional FAIL_TO_PASS-size filter for ATS studies.
    if MIN_FTP > 0:
        ftp = r.get("FAIL_TO_PASS") or []
        if isinstance(ftp, str):
            try:
                ftp = json.loads(ftp)
            except Exception:
                ftp = []
        if len(ftp) < MIN_FTP:
            continue
    r["docker_image"] = img
    r.pop("ip", None)
    try:
        ts = mts_swerebench(r)
        r["make_test_spec"] = json.dumps(ts.__dict__ if hasattr(ts, "__dict__") else ts,
                                         default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    except Exception as e:
        print(f"  skip {r.get('instance_id')}: make_test_spec failed: {type(e).__name__} {str(e)[:80]}")
        continue
    ps = r.get("problem_statement", "") or ""
    rows.append({
        "data_source": "swe",
        "prompt": [{"role": "system", "content": SWE_SYSTEM_PROMPT},
                   {"role": "user", "content": SWE_USER_PROMPT.format(problem_statement=ps)}],
        "ability": "swe",
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": json.dumps(r, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)),
    })
    picked.append(r["instance_id"])
    print(f"  [{len(rows)}/{need}] {r['instance_id']}  ({img.split('/')[-1]})")

if len(rows) < need:
    print(f"WARNING: only {len(rows)} usable tasks found (wanted {need})")

random.seed(0); random.shuffle(rows)
train_rows, val_rows = rows[:N_TRAIN], rows[N_TRAIN:N_TRAIN + N_VAL] or rows[:N_VAL]
pd.DataFrame(train_rows).to_parquet(os.path.join(OUT, "train_00000_verl.parquet"))
pd.DataFrame(val_rows).to_parquet(os.path.join(OUT, "test_00000_verl.parquet"))
print(f"\nwrote {len(train_rows)} train + {len(val_rows)} val tasks -> {OUT}")
print("train instances:", picked[:N_TRAIN])
