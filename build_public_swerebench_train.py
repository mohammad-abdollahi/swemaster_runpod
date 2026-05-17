#!/usr/bin/env python3
"""Faithful SWE-Master RL training data: the bundled swe-gym+rebench OOD demo
parquet IS the real training distribution (NOT SWE-bench, which is val-only).
Its docker_image fields point at BOSS Zhipin's private harbor registry; the
identical images are public on Docker Hub under `swerebench/`. Rewrite only
the image reference, keep prompt/reward_model/extra_info (incl make_test_spec).
"""
import json, os
import pandas as pd

SRC = "/home/temp/swemaster_runpod/data_examples/rl_data/ood_data/equal_0.3999999_0.9999999_swe-gym_and_rebench_demo_for_RL/train_00000_verl.parquet"
OUT = "/home/temp/swemaster_runpod/data_examples/rl_data/public_swerebench_train"
os.makedirs(OUT, exist_ok=True)

df = pd.read_parquet(SRC)

def rewrite(extra_info):
    d = json.loads(extra_info) if isinstance(extra_info, str) else dict(extra_info)
    img = d.get("docker_image", "")
    # harbor.weizhipin.com/arsenal-ai/swerebench/sweb.eval.x86_64.X[:tag]
    #  -> swerebench/sweb.eval.x86_64.X:latest  (public Docker Hub mirror)
    if "sweb.eval" in img:
        tail = img.split("/")[-1]
        if ":" in tail and tail.rsplit(":", 1)[0].endswith(tail.rsplit(":", 1)[0]):
            tail = tail  # keep as-is if already has the sweb.eval name
        name = tail.split(":")[0]
        d["docker_image"] = f"swerebench/{name}:latest"
    d.pop("ip", None)  # drop internal remote-docker host
    return json.dumps(d, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

df["extra_info"] = df["extra_info"].map(rewrite)
train = os.path.join(OUT, "train_00000_verl.parquet")
val = os.path.join(OUT, "test_00000_verl.parquet")
df.to_parquet(train); df.to_parquet(val)

for i, r in df.iterrows():
    d = json.loads(r["extra_info"])
    print(f"row {i}: instance={d.get('instance_id')}  image={d.get('docker_image')}  has_make_test_spec={'make_test_spec' in d}")
print("rows:", len(df), "-> wrote", train)
