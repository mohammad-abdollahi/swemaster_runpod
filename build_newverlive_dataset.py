#!/usr/bin/env python3
"""Build a NEW 64/16 dataset of SWE-Verified + SWE-Live instances (never trained on)
from nebius/SWE-rebench, in the exact verl parquet format (template prompt from verlive).
Images = swerebench/ namespace (pullable, runtime-compatible). 0 overlap with verlive."""
import pandas as pd, json, dataclasses
from datasets import load_dataset
from collections import defaultdict
from swebench_fork_swerebench.harness.test_spec.test_spec import make_test_spec as mts_swe

REPO_BENCH = json.load(open('/tmp/bench_repos.json'))
VERIFIED = set(REPO_BENCH['verified_repos'])
UNIVERSE = VERIFIED | set(REPO_BENCH['live_repos'])

def rows(p): return [ (json.loads(x) if isinstance(x,str) else x) for x in pd.read_parquet(p)['extra_info'] ]

# --- template (system msg + user wrapper) from an existing verlive row ---
tdf = pd.read_parquet('data_examples/rl_data/swerebench_verlive/train_00000_verl.parquet')
trow = tdf.iloc[0]
SYS_MSG = trow['prompt'][0]['content']
USER_TMPL = trow['prompt'][1]['content']
TMPL_PS = json.loads(trow['extra_info'])['problem_statement']
assert TMPL_PS in USER_TMPL, "template problem_statement not found in user message"
PREFIX, SUFFIX = USER_TMPL.split(TMPL_PS, 1)

# extra_info schema keys present in the local dataset (to mirror)
LOCAL_KEYS = list(json.loads(trow['extra_info']).keys())

# already-trained ids (exclude)
vl = rows('data_examples/rl_data/swerebench_verlive/train_00000_verl.parquet') + \
     rows('data_examples/rl_data/swerebench_verlive/test_00000_verl.parquet')
TRAINED = {x['instance_id'] for x in vl}

print("loading nebius/SWE-rebench ...")
d = load_dataset('nebius/SWE-rebench', split='test')

# candidates: repo in Verified|Live universe, not trained, has F2P + patch + test_patch + image
cand = []
for x in d:
    if x.get('repo') in UNIVERSE and x['instance_id'] not in TRAINED \
       and x.get('FAIL_TO_PASS') and x.get('patch') and x.get('test_patch') and x.get('docker_image'):
        cand.append(x)
print(f"candidates (Verified|Live, untrained, complete): {len(cand)}")

# diverse selection: round-robin across repos (cap dominance), keep some Verified repos in.
by_repo = defaultdict(list)
for x in sorted(cand, key=lambda r: r['instance_id']):
    by_repo[x['repo']].append(x)
# order repos: Verified first (so they're represented), then the rest, all sorted
repos_ordered = sorted([r for r in by_repo if r in VERIFIED]) + sorted([r for r in by_repo if r not in VERIFIED])
picked = []
ri = 0
while len(picked) < 80:
    progressed = False
    for r in repos_ordered:
        if by_repo[r]:
            picked.append(by_repo[r].pop(0)); progressed = True
            if len(picked) >= 80: break
    if not progressed: break
print(f"selected {len(picked)} instances across {len(set(p['repo'] for p in picked))} repos")
nver = sum(1 for p in picked if p['repo'] in VERIFIED)
print(f"  from SWE-Verified repos: {nver} | from SWE-Live repos: {len(picked)-nver}")

def to_verl_row(inst):
    """Build a verl row; returns None if make_test_spec can't be built for this instance."""
    inst = dict(inst)
    try:
        ts = mts_swe(inst)
        mts = json.dumps(dataclasses.asdict(ts) if dataclasses.is_dataclass(ts) else dict(ts.__dict__))
    except Exception:
        return None
    ei = {}
    for k in LOCAL_KEYS:
        if k == 'image_name':
            ei[k] = inst.get('docker_image')
        elif k == 'make_test_spec':
            ei[k] = mts
        else:
            ei[k] = inst.get(k, None)
    user = PREFIX + (inst.get('problem_statement') or '') + SUFFIX
    return {
        'data_source': 'swe',
        'prompt': [ {'role':'system','content':SYS_MSG}, {'role':'user','content':user} ],
        'ability': 'swe',
        'reward_model': {'style':'rule','ground_truth':''},
        'extra_info': json.dumps(ei),
    }

# convert + validate make_test_spec; keep only rows that build, until we have 80
valid_rows = []
for inst in picked + [x for r in repos_ordered for x in by_repo[r]]:  # picked first, then refill from leftovers
    if len(valid_rows) >= 80: break
    row = to_verl_row(inst)
    if row is not None:
        valid_rows.append(row)
print(f"valid rows (make_test_spec built): {len(valid_rows)}")
assert len(valid_rows) >= 80, f"only {len(valid_rows)} valid; need 80"
train_rows = valid_rows[:64]
val_rows   = valid_rows[64:80]
import os
out = 'data_examples/rl_data/swerebench_newverlive'
os.makedirs(out, exist_ok=True)
pd.DataFrame(train_rows).to_parquet(f'{out}/train_00000_verl.parquet')
pd.DataFrame(val_rows).to_parquet(f'{out}/test_00000_verl.parquet')
print(f"WROTE {out}: train={len(train_rows)} val={len(val_rows)}")
# sanity: reload + verify schema matches local + 0 overlap
nt = pd.read_parquet(f'{out}/train_00000_verl.parquet')
assert list(nt.columns) == list(tdf.columns), f"column mismatch {list(nt.columns)} vs {list(tdf.columns)}"
new_ids = {json.loads(x)['instance_id'] for x in pd.concat([nt, pd.read_parquet(f'{out}/test_00000_verl.parquet')])['extra_info']}
print(f"columns OK | overlap with trained: {len(new_ids & TRAINED)} (0=good)")
print("sample new train ids:", [json.loads(x)['instance_id'] for x in nt['extra_info'].head(5)])
print("sample repos:", sorted({json.loads(x)['repo'] for x in nt['extra_info']})[:12])
