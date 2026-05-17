#!/usr/bin/env python3
"""Build a verl parquet for the original SWE-Master RL pipeline using a
SWE-bench Verified instance whose docker image is PUBLIC on Docker Hub.

The in-repo demo records all point at BOSS Zhipin's private
harbor.weizhipin.com registry (unpullable outside their network). The
instance itself (astropy__astropy-12907) is genuine SWE-bench Verified, so
its image exists publicly as swebench/sweb.eval.x86_64.<id>:latest.
Schema matches DeepSWE_RL/rllm/scripts/data/swe_dataset.py exactly.
"""
import json, os, sys
import pandas as pd
sys.path.insert(0, "/home/temp/swemaster_runpod/DeepSWE_RL/rllm")
from rllm.agents.system_prompts import SWE_SYSTEM_PROMPT, SWE_USER_PROMPT

SRC = "/home/temp/swemaster_runpod/data_examples/inference_data/swe_bench_verified/test_swe_bench_verified_full_ip_demo.json"
OUT_DIR = "/home/temp/swemaster_runpod/data_examples/rl_data/public_swebench_verified"
os.makedirs(OUT_DIR, exist_ok=True)

rec = json.load(open(SRC))[0]
iid = rec["instance_id"]                       # astropy__astropy-12907
pub = "swebench/sweb.eval.x86_64." + iid.replace("__", "_1776_") + ":latest"
rec["docker_image"] = pub                      # rewrite private -> public
rec.pop("ip", None)                            # drop internal remote-docker ip

problem = rec.get("problem_statement", "")
row = {
    "data_source": "swe",
    "prompt": [
        {"role": "system", "content": SWE_SYSTEM_PROMPT},
        {"role": "user", "content": SWE_USER_PROMPT.format(problem_statement=problem)},
    ],
    "ability": "swe",
    "reward_model": {"style": "rule", "ground_truth": ""},
    "extra_info": json.dumps(rec),
}
df = pd.DataFrame([row])
train = os.path.join(OUT_DIR, "train_00000_verl.parquet")
val = os.path.join(OUT_DIR, "test_00000_verl.parquet")
df.to_parquet(train); df.to_parquet(val)
print("instance:", iid)
print("public image:", pub)
print("prompt tokens(approx chars/4):", (len(SWE_SYSTEM_PROMPT)+len(problem))//4)
print("wrote:", train, "and", val, "rows=", len(df))
