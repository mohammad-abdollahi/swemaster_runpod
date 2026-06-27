#!/usr/bin/env python
"""
Validate ApptainerRuntime end-to-end WITHOUT vLLM/GPU, on a login node.

For a single instance it:
  1. boots the Apptainer container (sif + overlay + fakeroot),
  2. runs setup_env (inherited swebench setup),
  3. applies the GOLD patch (ds['patch']) -> the fix the model is supposed to find,
  4. grades via the inherited swebench reward path (runs /run_tests.sh + parses).

Expected: reward == 1 (the gold patch resolves the instance). If this passes, the
grading half of the harness is correct and only the agent rollout (vLLM) remains.

Usage:
  source env.sh && python scripts/test_runtime.py <instance_id>
"""
import os
import sys
from datasets import load_from_disk

sys.path.insert(0, os.path.join(os.environ["R2EGYM_DIR"], "src"))
from r2egym.agenthub.runtime.apptainer import ApptainerRuntime  # noqa: E402


def main():
    inst = sys.argv[1] if len(sys.argv) > 1 else "astropy__astropy-12907"
    ds = load_from_disk(os.environ["DATASET_DIR"])
    row = next(r for r in ds if r["instance_id"] == inst)
    print(f"== instance {inst} | repo {row['repo']} | image {row['docker_image']}")

    rt = ApptainerRuntime(ds=row)
    try:
        print("-- container up; sanity command --")
        out, rc = rt.run("python -c 'import sys; print(sys.version)'")
        print("python:", out.strip(), "rc=", rc)

        out, rc = rt.run("git -C /testbed rev-parse HEAD")
        print("HEAD:", out.strip(), "rc=", rc)

        print("-- applying GOLD patch --")
        out, rc = rt.apply_patch(row["patch"])
        print("git apply rc=", rc, "| out:", out[:300])

        print("-- grading (runs /run_tests.sh + swebench parser) --")
        reward, test_out = rt._calculate_reward(get_test_output=True, timeout=900)
        print("=" * 60)
        print(f"REWARD = {reward}   (1 => gold patch resolves instance => harness OK)")
        print("=" * 60)
        print("test output tail:\n", test_out[-1500:])
    finally:
        rt.close()


if __name__ == "__main__":
    main()
