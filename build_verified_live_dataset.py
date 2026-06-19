#!/usr/bin/env python3
"""Build a Verified+Live SWE-rebench training set (user request 2026-06-18).

Pool = SWE-rebench instances whose repo appears in EITHER overlap tsv:
  - live_swerebench_overlap.tsv      (SWE-bench Live  ∩ SWE-rebench, 83 repos)
  - verified_swerebench_overlap.tsv  (SWE-bench Verified ∩ SWE-rebench, 7 repos)

So the dataset spans recognizable benchmark repos common to SWE-rebench and the two
SWE-bench tracks. The earlier "verified images absent" exclusion was wrong — all the
verified-repo `swerebench/` images (astropy, matplotlib, sphinx, sympy, xarray) are
pullable (seaborn/scikit-learn have none, so they drop out naturally).

Conversion mirrors build_live_overlap_dataset.py (make_test_spec via swerebench fork).
Output: data_examples/rl_data/swerebench_verlive/{train,test}_00000_verl.parquet
        + verlive_manifest.tsv (provenance) + images_to_pull.txt.

Env knobs: N_TRAIN(64) N_VAL(16) MIN_FTP(1) MAX_FTP(40) MAX_P2P(160)
           TRAIN_PER_REPO(3) VAL_PER_REPO(1) SEED(0) OUT_DIR.
"""
import json, os, sys, random, collections
import pandas as pd

sys.path.insert(0, "/home/temp/swemaster_runpod/DeepSWE_RL/rllm")
sys.path.insert(0, "/home/temp/swemaster_runpod/R2E-Gym/src")
from datasets import load_dataset
from rllm.agents.system_prompts import SWE_SYSTEM_PROMPT, SWE_USER_PROMPT
from swebench_fork_swerebench.harness.test_spec.test_spec import make_test_spec as mts_swerebench

ROOT = "/home/temp/swemaster_runpod"
OUT = os.environ.get("OUT_DIR", os.path.join(ROOT, "data_examples/rl_data/swerebench_verlive"))
N_TRAIN = int(os.environ.get("N_TRAIN", "64"))
N_VAL = int(os.environ.get("N_VAL", "16"))
# Per-type F2P floor: keep ALL verified (even single-test, to preserve representation),
# but require LIVE instances to be multi-test so R22 has tests to gate (hybrid build).
MIN_FTP_VERIFIED = int(os.environ.get("MIN_FTP_VERIFIED", "1"))
MIN_FTP_LIVE = int(os.environ.get("MIN_FTP_LIVE", "3"))
MAX_FTP = int(os.environ.get("MAX_FTP", "40"))
MAX_P2P = int(os.environ.get("MAX_P2P", "160"))   # keep terminal /run_tests.sh fast
TRAIN_PER_REPO = int(os.environ.get("TRAIN_PER_REPO", "3"))
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


def load_overlap_repos(fname):
    """Both tsvs share columns: bare_name<TAB>swe_bench_*_repo<TAB>swe_rebench_repo."""
    repos = set()
    with open(os.path.join(ROOT, fname)) as fh:
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
    live = load_overlap_repos("live_swerebench_overlap.tsv")
    verified = load_overlap_repos("verified_swerebench_overlap.tsv")
    pool_repos = live | verified
    print(f"overlap repos: live={len(live)} verified={len(verified)} union={len(pool_repos)}")
    print(f"verified repos: {sorted(verified)}")

    print("loading nebius/SWE-rebench (cached) ...")
    ds = load_dataset("nebius/SWE-rebench", split="test")
    pool = collections.defaultdict(list)   # repo -> [(ps_len, iid, row, is_verified)]
    for i in range(len(ds)):
        r = ds[i]
        repo = r.get("repo")
        if repo not in pool_repos:
            continue
        img = r.get("docker_image") or r.get("image_name") or ""
        if "swerebench/" not in img:
            continue
        nf = len(plist(r.get("FAIL_TO_PASS")))
        npp = len(plist(r.get("PASS_TO_PASS")))
        minf = MIN_FTP_VERIFIED if repo in verified else MIN_FTP_LIVE
        if not (minf <= nf <= MAX_FTP and 1 <= npp <= MAX_P2P):
            continue
        iid = r.get("instance_id")
        pool[repo].append((len(r.get("problem_statement") or ""), iid, dict(r), repo in verified))
    for repo in pool:
        pool[repo].sort(key=lambda x: x[0])   # shortest problem statement first (tractable)
    # order: verified repos FIRST (guarantee representation), then by pool size
    repos_sorted = sorted(pool, key=lambda rp: (rp not in verified, -len(pool[rp]), rp))
    nverif = sum(len(pool[r]) for r in pool if r in verified)
    print(f"candidate pool: {sum(len(v) for v in pool.values())} instances across {len(pool)} "
          f"repos ({nverif} verified-repo instances, {len([r for r in pool if r in verified])} verified repos)")

    used = set()

    def round_robin(n, per_repo):
        picked, taken = [], collections.Counter()
        progress = True
        while len(picked) < n and progress:
            progress = False
            for repo in repos_sorted:
                if len(picked) >= n:
                    break
                if taken[repo] >= per_repo:
                    continue
                for ps_len, iid, row, isv in pool[repo]:
                    if iid in used:
                        continue
                    picked.append((repo, iid, row, isv))
                    used.add(iid)
                    taken[repo] += 1
                    progress = True
                    break
        return picked

    train_pick = round_robin(N_TRAIN, TRAIN_PER_REPO)
    val_pick = round_robin(N_VAL, VAL_PER_REPO)
    print(f"selected: {len(train_pick)} train / {len(val_pick)} val")
    if len(train_pick) < N_TRAIN or len(val_pick) < N_VAL:
        print(f"WARNING wanted {N_TRAIN}/{N_VAL}, got {len(train_pick)}/{len(val_pick)}; "
              f"relax MAX_P2P / per-repo caps / MIN_FTP.")

    tv = sum(1 for *_x, isv in train_pick if isv)
    vv = sum(1 for *_x, isv in val_pick if isv)
    print(f"  train: {tv} verified-repo / {len(train_pick)-tv} live-repo")
    print(f"  val:   {vv} verified-repo / {len(val_pick)-vv} live-repo")

    train_rows = [make_row(row) for _, _, row, _ in train_pick]
    val_rows = [make_row(row) for _, _, row, _ in val_pick]
    rng = random.Random(SEED)
    rng.shuffle(train_rows)

    train_ids = set(json.loads(r["extra_info"])["instance_id"] for r in train_rows)
    val_ids = set(json.loads(r["extra_info"])["instance_id"] for r in val_rows)
    leak = train_ids & val_ids
    assert not leak, f"LEAKAGE: {leak}"

    pd.DataFrame(train_rows).to_parquet(os.path.join(OUT, "train_00000_verl.parquet"))
    pd.DataFrame(val_rows).to_parquet(os.path.join(OUT, "test_00000_verl.parquet"))

    with open(os.path.join(OUT, "verlive_manifest.tsv"), "w") as fh:
        fh.write("split\tsource\trepo\tinstance_id\tdocker_image\n")
        for repo, iid, row, isv in train_pick:
            fh.write(f"train\t{'verified' if isv else 'live'}\t{repo}\t{iid}\t{row.get('docker_image','')}\n")
        for repo, iid, row, isv in val_pick:
            fh.write(f"val\t{'verified' if isv else 'live'}\t{repo}\t{iid}\t{row.get('docker_image','')}\n")

    with open(os.path.join(OUT, "images_to_pull.txt"), "w") as fh:
        for _, _, row, _ in train_pick + val_pick:
            img = row.get("docker_image") or row.get("image_name")
            if ":" not in img.split("/")[-1]:
                img += ":latest"
            fh.write(img + "\n")

    print(f"\nwrote {len(train_rows)} train + {len(val_rows)} val -> {OUT}")
    print(f"manifest: {OUT}/verlive_manifest.tsv | images: {OUT}/images_to_pull.txt")


if __name__ == "__main__":
    main()
