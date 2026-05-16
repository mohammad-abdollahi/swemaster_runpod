"""Build one conda env per SWE-bench demo instance (Docker-free).

Mirrors the ``make_test_spec`` env/repo scripts but targets the repos we
already cloned at /scratch/.../repos and a conda env per instance at
/scratch/.../test_envs/<instance_id>.  Idempotent: a ``.env_ready`` sentinel
skips already-built envs.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import swe_bench_demo  # noqa: E402
from swe_sandbox import _repo_dir_for  # noqa: E402
from swe_test_reward import TEST_ENVS_ROOT, _as_list  # noqa: E402

import pandas as pd  # noqa: E402

CONDA = os.environ.get("CONDA_EXE", "/opt/conda/bin/conda")


def sh(cmd: str, cwd: str | None = None, timeout: int = 1800) -> tuple[int, str]:
    p = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True,
                        text=True, timeout=timeout)
    return p.returncode, (p.stdout + "\n" + p.stderr)


def build_one(row: dict) -> bool:
    iid = row["instance_id"]
    spec = row["make_test_spec"]
    if isinstance(spec, str):
        spec = json.loads(spec)
    cfg = spec.get("install_config", {})
    py = cfg.get("python", "3.9")
    env_dir = os.path.join(TEST_ENVS_ROOT, iid)
    sentinel = os.path.join(env_dir, ".env_ready")
    if os.path.isfile(sentinel):
        print(f"[skip] {iid} (already built)")
        return True

    repo_dir = _repo_dir_for(row["repo"])
    if not os.path.isdir(repo_dir):
        print(f"[FAIL] {iid}: repo not cloned at {repo_dir}")
        return False

    print(f"[build] {iid}  python={py}  repo={repo_dir}")
    os.makedirs(TEST_ENVS_ROOT, exist_ok=True)

    # 1. conda env
    rc, out = sh(f"{CONDA} create -p {env_dir} python={py} -y -q")
    if rc != 0:
        print(f"[FAIL] {iid}: conda create\n{out[-800:]}")
        return False
    env_py = os.path.join(env_dir, "bin", "python")

    # 2. ensure base commit (sandbox may have left edits from a prior rollout)
    sh(f"git reset --hard {row['base_commit']} && git clean -fdx", cwd=repo_dir)

    # 3. requirements: prefer the in-repo requirements file the spec names.
    reqs = _as_list(cfg.get("reqs_path")) or ["requirements.txt"]
    for rp in reqs:
        rp_full = os.path.join(repo_dir, rp)
        if os.path.isfile(rp_full):
            rc, out = sh(f"{env_py} -m pip install -q -r {rp}", cwd=repo_dir)
            print(f"  reqs {rp}: rc={rc}")

    # 4. pip_packages (pytest etc.)
    for pp in _as_list(cfg.get("pip_packages")):
        sh(f"{env_py} -m pip install -q {pp}", cwd=repo_dir)

    # 5. project install (e.g. ``pip install -e .[dev]``)
    install_cmd = cfg.get("install", "pip install -e .")
    install_cmd = install_cmd.replace("pip install",
                                      f"{env_py} -m pip install -q", 1)
    rc, out = sh(install_cmd, cwd=repo_dir, timeout=2400)
    if rc != 0:
        print(f"[warn] {iid}: project install rc={rc} (continuing)\n{out[-600:]}")

    # 6. sanity: pytest importable
    rc, out = sh(f"{env_py} -m pytest --version", cwd=repo_dir)
    if rc != 0:
        sh(f"{env_py} -m pip install -q pytest", cwd=repo_dir)

    Path(sentinel).write_text("ok\n")
    print(f"[done] {iid}")
    return True


def main() -> None:
    parquet = os.environ.get("SWE_RL_PARQUET")
    df = pd.read_parquet(parquet) if parquet else None
    if df is None:
        # Fall back to the default demo parquet via data.py path resolution.
        ds = swe_bench_demo()
        rows = [
            {"instance_id": ds["instance_id"][i],
             "repo": ds["repo"][i],
             "base_commit": ds["base_commit"][i],
             "make_test_spec": None}
            for i in range(len(ds))
        ]
        # data.py doesn't carry make_test_spec; re-read parquet directly.
        import glob
        cands = glob.glob("data_examples/rl_data/**/train_00000.parquet",
                          recursive=True)
        df = pd.read_parquet(cands[0])
    ok = 0
    for _, r in df.iterrows():
        if build_one(r.to_dict()):
            ok += 1
    print(f"[build_test_envs] {ok}/{len(df)} envs ready")


if __name__ == "__main__":
    main()
