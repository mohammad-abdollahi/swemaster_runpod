#!/usr/bin/env python3
"""Build the ATS-RL "live64" training set.

Design (per user, 2026-06-16):
  - Core: keep the existing, validated, locally-cached `swerebench_min_ftp5_v2`
    train instances (39 obscure-but-cached repos).
  - Add: ~25 instances drawn from SWE-rebench rows whose repo ALSO appears in
    SWE-bench *Live* (live_swerebench_overlap.tsv) — recognizable projects with
    public `swerebench/` Docker Hub images.
  - Val: 16 *disjoint* live-overlap instances (fixes the v2 leakage where the
    val split was literally the first 8 train rows).
  - Final: 64 train / 16 val, sized for ~10 epochs of batch-16 RL.

Verified-overlap repos (astropy/sympy/xarray/...) are intentionally excluded:
their SWE-rebench images are essentially absent from the public namespace
(1 usable instance total) — see verified_swerebench_overlap.tsv analysis.

Output: data_examples/rl_data/swerebench_live64/{train,test}_00000_verl.parquet
plus a provenance manifest live64_manifest.tsv.

Env knobs: N_ADD (default 25), N_VAL (16), MAX_P2P (160), MIN_FTP (5),
MAX_FTP (40), TRAIN_PER_REPO (2), VAL_PER_REPO (1), OUT_DIR, SEED (0).
"""
import json, os, sys, random, collections
import pandas as pd
sys.path.insert(0, "/home/temp/swemaster_runpod/DeepSWE_RL/rllm")
sys.path.insert(0, "/home/temp/swemaster_runpod/R2E-Gym/src")
from datasets import load_dataset
from rllm.agents.system_prompts import SWE_SYSTEM_PROMPT, SWE_USER_PROMPT
from swebench_fork_swerebench.harness.test_spec.test_spec import make_test_spec as mts_swerebench

ROOT = "/home/temp/swemaster_runpod"
V2_DIR = os.path.join(ROOT, "data_examples/rl_data/swerebench_min_ftp5_v2")
OUT = os.environ.get("OUT_DIR", os.path.join(ROOT, "data_examples/rl_data/swerebench_live64"))
N_ADD = int(os.environ.get("N_ADD", "25"))
N_VAL = int(os.environ.get("N_VAL", "16"))
MIN_FTP = int(os.environ.get("MIN_FTP", "5"))
MAX_FTP = int(os.environ.get("MAX_FTP", "40"))
MAX_P2P = int(os.environ.get("MAX_P2P", "160"))   # keep terminal /run_tests.sh fast
TRAIN_PER_REPO = int(os.environ.get("TRAIN_PER_REPO", "2"))
VAL_PER_REPO = int(os.environ.get("VAL_PER_REPO", "1"))
SEED = int(os.environ.get("SEED", "0"))
os.makedirs(OUT, exist_ok=True)


def plist(v):
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return []
    return []


def load_live_repos():
    repos = set()
    with open(os.path.join(ROOT, "live_swerebench_overlap.tsv")) as fh:
        for line in fh:
            if line.startswith(("#", "bare_name")) or not line.strip() or "CHECK" in line:
                continue
            p = line.split("\t")
            if len(p) >= 3:
                repos.add(p[2].split("#")[0].strip())
    return repos


def make_row(r):
    img = r.get("docker_image") or r.get("image_name")
    if ":" not in img.split("/")[-1]:
        img = img + ":latest"
    r = dict(r)
    r["docker_image"] = img
    r.pop("ip", None)
    ts = mts_swerebench(r)
    r["make_test_spec"] = json.dumps(
        ts.__dict__ if hasattr(ts, "__dict__") else ts,
        default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o),
    )
    ps = r.get("problem_statement", "") or ""
    return {
        "data_source": "swe",
        "prompt": [
            {"role": "system", "content": SWE_SYSTEM_PROMPT},
            {"role": "user", "content": SWE_USER_PROMPT.format(problem_statement=ps)},
        ],
        "ability": "swe",
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": json.dumps(r, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)),
    }


def main():
    live = load_live_repos()
    print(f"live-overlap repos: {len(live)}")

    # --- core: existing v2 train (39 rows, cached images) ---
    v2 = pd.read_parquet(os.path.join(V2_DIR, "train_00000_verl.parquet"))
    core_ids = set(json.loads(r)["instance_id"] for r in v2["extra_info"])
    print(f"v2 core train rows: {len(v2)}")

    # --- candidate pool from SWE-rebench live-overlap ---
    print("loading nebius/SWE-rebench (cached) ...")
    ds = load_dataset("nebius/SWE-rebench", split="test")
    pool = collections.defaultdict(list)  # repo -> [(ps_len, iid, row)]
    for i in range(len(ds)):
        r = ds[i]
        repo = r.get("repo")
        if repo not in live:
            continue
        img = r.get("docker_image") or ""
        if "swerebench/" not in img:
            continue
        nf = len(plist(r.get("FAIL_TO_PASS")))
        npp = len(plist(r.get("PASS_TO_PASS")))
        if not (MIN_FTP <= nf <= MAX_FTP and 1 <= npp <= MAX_P2P):
            continue
        iid = r.get("instance_id")
        if iid in core_ids:
            continue
        pool[repo].append((len(r.get("problem_statement") or ""), iid, dict(r)))
    for repo in pool:
        pool[repo].sort(key=lambda x: x[0])  # shortest problem statement first (tractable)
    repos_sorted = sorted(pool, key=lambda rp: (-len(pool[rp]), rp))
    print(f"candidate pool: {sum(len(v) for v in pool.values())} instances across {len(pool)} repos")

    # --- round-robin selection: train first (<=TRAIN_PER_REPO/repo), then val (<=VAL_PER_REPO) ---
    used = set()

    def round_robin(n, per_repo):
        picked = []
        taken_per = collections.Counter()
        progress = True
        while len(picked) < n and progress:
            progress = False
            for repo in repos_sorted:
                if len(picked) >= n:
                    break
                if taken_per[repo] >= per_repo:
                    continue
                for ps_len, iid, row in pool[repo]:
                    if iid in used:
                        continue
                    picked.append((repo, iid, row))
                    used.add(iid)
                    taken_per[repo] += 1
                    progress = True
                    break
        return picked

    train_add = round_robin(N_ADD, TRAIN_PER_REPO)
    val_pick = round_robin(N_VAL, VAL_PER_REPO)
    print(f"selected: +{len(train_add)} train  /  {len(val_pick)} val")

    if len(train_add) < N_ADD or len(val_pick) < N_VAL:
        print(f"WARNING: wanted +{N_ADD}/{N_VAL}, got +{len(train_add)}/{len(val_pick)}; "
              f"relax MAX_P2P or per-repo caps.")

    # --- build rows ---
    add_rows = [make_row(row) for _, _, row in train_add]
    val_rows = [make_row(row) for _, _, row in val_pick]

    train_rows = list(v2.to_dict("records")) + add_rows
    rng = random.Random(SEED)
    rng.shuffle(train_rows)

    # --- leakage guard ---
    train_ids = set(json.loads(r["extra_info"])["instance_id"] for r in train_rows)
    val_ids = set(json.loads(r["extra_info"])["instance_id"] for r in val_rows)
    leak = train_ids & val_ids
    assert not leak, f"LEAKAGE: {leak}"

    pd.DataFrame(train_rows).to_parquet(os.path.join(OUT, "train_00000_verl.parquet"))
    pd.DataFrame(val_rows).to_parquet(os.path.join(OUT, "test_00000_verl.parquet"))

    # --- provenance manifest ---
    with open(os.path.join(OUT, "live64_manifest.tsv"), "w") as fh:
        fh.write("split\tsource\trepo\tinstance_id\tdocker_image\n")
        for r in v2["extra_info"]:
            e = json.loads(r)
            fh.write(f"train\tv2_core\t{e['repo']}\t{e['instance_id']}\t{e.get('docker_image','')}\n")
        for repo, iid, row in train_add:
            fh.write(f"train\tlive_overlap\t{repo}\t{iid}\t{row.get('docker_image','')}\n")
        for repo, iid, row in val_pick:
            fh.write(f"val\tlive_overlap\t{repo}\t{iid}\t{row.get('docker_image','')}\n")

    # images to pre-pull (the new live-overlap ones; v2 already cached)
    with open(os.path.join(OUT, "images_to_pull.txt"), "w") as fh:
        for _, _, row in train_add + val_pick:
            img = row.get("docker_image")
            if ":" not in img.split("/")[-1]:
                img += ":latest"
            fh.write(img + "\n")

    print(f"\nwrote {len(train_rows)} train + {len(val_rows)} val -> {OUT}")
    print(f"manifest: {OUT}/live64_manifest.tsv")
    print(f"images_to_pull: {OUT}/images_to_pull.txt ({len(train_add)+len(val_pick)} new images)")


if __name__ == "__main__":
    main()
