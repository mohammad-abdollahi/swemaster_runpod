#!/usr/bin/env python3
"""Block-A shared prerequisite: the PROBE PASS.

Per IMPLEMENTATION_PLAN.md §4, every Block-A test-selection arm (S2 variance,
S1 Goldilocks, S11 token-efficiency) is a different ranking/filter over ONE
per-instance probe CSV. This script produces that CSV.

The probe records, for each instance, the outcome of running the (frozen) base
policy `n` times: the partial-credit pass-fraction and total reward per rollout,
plus token cost. Two input modes:

  --from-log LOGFILE     Harvest the `!!!!!!! ATS terminal:` lines emitted by a
                         training/eval run. RELIABLE only if the line carries
                         `instance_id=` (added 2026-06-16 to swe_arsenal.py);
                         older logs fall back to docker-image pairing (noisy
                         under rollout concurrency — a warning is printed).

  --synth N              Emit a synthetic probe CSV for N instances (for unit-
                         testing the selection scripts without a GPU run).

Output CSV columns (the contract every select_*.py depends on):
  instance_id, n_rollouts,
  frac_solved_mean, frac_solved_list,      # partial-credit pass fraction in [0,1]
  reward_mean, reward_var, reward_list,     # ATS `total` per rollout
  prompt_tokens_mean, completion_tokens_mean, tokens_mean   # cost (approx ok)

Usage:
  python probe_instances.py --from-log /home/temp/control_live64.log \
      --out block_a_selection/probe_live64.csv
  python probe_instances.py --synth 64 --out /tmp/probe_synth.csv
"""
from __future__ import annotations
import argparse, csv, json, re, statistics as st, sys
from collections import defaultdict
from pathlib import Path

# instance_id-tagged terminal line (preferred, reliable under concurrency)
ATS_TAGGED = re.compile(
    r"ATS terminal: instance_id=(?P<iid>\S+) "
    r"frac_solved=(?P<frac>[-0-9.eE]+) "
    r"r_solve=(?P<r_solve>[-0-9.eE]+) "
    r"frac_p2p_broken=(?P<fp2p>[-0-9.eE]+) "
    r"r_regress=(?P<r_regress>[-0-9.eE]+) "
    r"r_token=(?P<r_token>[-0-9.eE]+) "
    r"dense_sum=(?P<dense>[-0-9.eE]+) "
    r"total=(?P<total>[-0-9.eE]+) "
)
# legacy untagged line (instance_id must be paired from a preceding docker line)
ATS_UNTAGGED = re.compile(r"ATS terminal: frac_solved=(?P<frac>[-0-9.eE]+).*?total=(?P<total>[-0-9.eE]+)")
DOCKER_RE = re.compile(r"swerebench/sweb\.eval\.x86_64\.([A-Za-z0-9_.\-]+?)(?::latest|\s|$)")
# token-ref used by the reward so we can back out approx tokens from r_token if needed
TOKEN_REF = 32000.0
LAM_TOKEN = 0.1


def harvest_log(path: Path):
    """Return {instance_id: {'frac':[...], 'total':[...], 'tokens':[...]}}."""
    recs = defaultdict(lambda: {"frac": [], "total": [], "tokens": []})
    tagged = untagged = 0
    last_iid = None
    for line in path.read_text(errors="ignore").splitlines():
        # track most-recent docker image as fallback instance id
        dm = DOCKER_RE.search(line)
        if dm:
            # swerebench encodes owner_1776_repo-num; convert to swebench id owner__repo-num
            raw = dm.group(1)
            last_iid = raw.replace("_1776_", "__")
        m = ATS_TAGGED.search(line)
        if m:
            tagged += 1
            iid = m.group("iid")
            frac = float(m.group("frac"))
            total = float(m.group("total"))
            r_token = float(m.group("r_token"))
            tokens = (-r_token / LAM_TOKEN) * TOKEN_REF if r_token < 0 else 0.0
            recs[iid]["frac"].append(frac)
            recs[iid]["total"].append(total)
            recs[iid]["tokens"].append(tokens)
            continue
        m = ATS_UNTAGGED.search(line)
        if m and last_iid:
            untagged += 1
            recs[last_iid]["frac"].append(float(m.group("frac")))
            recs[last_iid]["total"].append(float(m.group("total")))
            recs[last_iid]["tokens"].append(0.0)
    if tagged == 0 and untagged > 0:
        print(f"[warn] no instance_id-tagged terminal lines; used docker-image "
              f"pairing for {untagged} rollouts (NOISY under concurrency).", file=sys.stderr)
    print(f"[probe] harvested {tagged} tagged + {untagged} untagged rollouts "
          f"over {len(recs)} instances", file=sys.stderr)
    return recs


def synth(n):
    """Deterministic synthetic probe spanning the difficulty/variance spectrum."""
    recs = {}
    for i in range(n):
        # spread mean solve fraction 0..1; variance peaks in the middle band
        p = (i + 0.5) / n
        # 4 rollouts: bracket p so variance is realistic
        spread = 0.4 * (1 - abs(p - 0.5) * 2)  # high in middle, 0 at extremes
        fracs = [max(0.0, min(1.0, p - spread)), max(0.0, min(1.0, p)),
                 max(0.0, min(1.0, p)), max(0.0, min(1.0, p + spread))]
        totals = [10.0 * f - 0.05 for f in fracs]
        toks = 8000 + i * 200  # cost grows with index
        recs[f"synthrepo__inst-{i}"] = {"frac": fracs, "total": totals,
                                        "tokens": [toks] * 4}
    return recs


def write_csv(recs, out):
    rows = []
    for iid, d in sorted(recs.items()):
        fr = d["frac"]
        tot = d["total"]
        tok = [t for t in d["tokens"] if t > 0] or [0.0]
        if not fr:
            continue
        rows.append({
            "instance_id": iid,
            "n_rollouts": len(fr),
            "frac_solved_mean": round(st.mean(fr), 4),
            "frac_solved_list": json.dumps([round(x, 4) for x in fr]),
            "reward_mean": round(st.mean(tot), 4),
            "reward_var": round(st.pvariance(tot) if len(tot) > 1 else 0.0, 6),
            "reward_list": json.dumps([round(x, 4) for x in tot]),
            "prompt_tokens_mean": round(st.mean(tok), 1),
            "completion_tokens_mean": 0.0,
            "tokens_mean": round(st.mean(tok), 1),
        })
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[probe] wrote {len(rows)} instances -> {out}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-log")
    ap.add_argument("--synth", type=int)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.synth:
        recs = synth(a.synth)
    elif a.from_log:
        recs = harvest_log(Path(a.from_log))
    else:
        ap.error("need --from-log or --synth")
    write_csv(recs, a.out)


if __name__ == "__main__":
    main()
