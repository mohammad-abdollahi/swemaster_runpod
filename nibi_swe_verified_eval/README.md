# SWE-Master-4B-SFT × SWE-bench Verified — Docker-free eval on Nibi

Evaluate [`RUC-AIBOX/SWE-Master-4B-SFT`](https://huggingface.co/RUC-AIBOX/SWE-Master-4B-SFT)
on **SWE-bench Verified** (500 instances, pass@1) on the **Nibi** Alliance Canada cluster,
which has **no Docker**. We reproduce the model's official pipeline (vLLM + the
[R2E-Gym](https://github.com/R2E-Gym/R2E-Gym) agent harness) and replace the Docker
execution backend with **Apptainer**.

## Why this is non-trivial here
- **No Docker.** R2E-Gym ships only Docker/Kubernetes runtimes. We add an
  `ApptainerRuntime` (see below).
- **Compute nodes are offline.** Everything (model, dataset, container images) is
  pre-staged on the login node.
- **Inode quota is the binding limit** (`/project` ~5K free, `/scratch` was at its cap).
  So: container images are single-file `.sif`, the writable layer is a single-file
  overlay, and the venv lives on `/scratch`.

## Architecture
```
 vLLM server (swebench_pro/.venv, GPU)  ──OpenAI API──▶  R2E-Gym agent (agent venv, CPU)
 serves SWE-Master-4B-SFT                                 openhands scaffold, 150 steps
        on 1×H100                                                │
                                                                 ▼
                                         ApptainerRuntime: per-instance container
                                         apptainer exec --fakeroot --overlay <img.overlay>
                                                          slimshetty/swebench-verified:<inst>.sif
                                         (agent edits /testbed, runs tests; grading via swebench parser)
```

### The ApptainerRuntime
`swemaster_runpod/R2E-Gym/src/r2egym/agenthub/runtime/apptainer.py` subclasses
`DockerRuntime`, inheriting **all** swebench grading logic (`make_test_spec`,
`get_logs_eval`, `_calculate_reward_*`). It overrides only the container primitives:

| concern | Docker | Apptainer (here) |
|---|---|---|
| image | `docker pull` + daemon | `.sif` built from `docker://slimshetty/...` |
| writable, root, persistent FS | container layer | `--fakeroot --overlay <single-file img>` |
| run a command | `container.exec_run` | `apptainer exec --fakeroot --overlay … bash -c` |
| copy file / patch in | `put_archive` (tar) | stream over stdin (`cat >`, `git apply -`) |
| reset / close | restart container | recreate / delete overlay file |

Key facts validated on Nibi: there is **no `/etc/subuid`** entry, so `--fakeroot` uses a
root-mapped namespace + fuse-overlayfs. The overlay file **must be created with
`--fakeroot`** (`apptainer overlay create --fakeroot --size …`) or the upper dir is
not writable. `backend="apptainer"` is dispatched in `environment/env.py`.

## Layout
```
evaluation/
├── env.sh                      # all paths + cache/quota-aware env (source everywhere)
├── requirements_agent.txt      # curated R2E-Gym deps (no Vertex/google-cloud -> no pip backtracking)
├── server_eval.sh              # SLURM job: vLLM + agent, smoke or full
├── scripts/
│   ├── 01_setup_agent_venv.sh  # agent venv on /scratch (login node)
│   ├── 02_stage_model.sh       # download SWE-Master-4B-SFT snapshot (login node)
│   ├── 03_stage_dataset.sh     # save R2E-Gym/SWE-Bench-Verified to disk (login node)
│   ├── build_sifs.sh           # pre-build per-instance .sif (login node, online)
│   ├── serve_vllm.sh           # vLLM launcher (uses swebench_pro/.venv)
│   ├── run_eval.sh             # the R2E-Gym runagent_multiple invocation
│   ├── test_runtime.py         # validate grading w/o GPU (apply gold patch -> reward==1)
│   └── report.py               # resolve-rate@1 from the trajectory jsonl
```
Heavy artifacts live under `$WORK=/scratch/moham98/swemaster` (venv, models, dataset,
sif, overlays, apptainer_cache, results, logs).

## Run it
```bash
# --- one-time, on the LOGIN NODE (online) ---
cd /project/6087623/mohammadA/evaluation
bash scripts/01_setup_agent_venv.sh
bash scripts/02_stage_model.sh
bash scripts/03_stage_dataset.sh
bash scripts/build_sifs.sh 0 5            # smoke set; use 0 500 for the full run

# --- validate grading with no GPU (login node) ---
source env.sh && source $VENV/bin/activate
python scripts/test_runtime.py astropy__astropy-12907   # expect REWARD = 1

# --- smoke run on a GPU node ---
sbatch server_eval.sh 0 5 2 smoke
python scripts/report.py $WORK/results/smoke/smoke.jsonl

# --- full 500 (build all SIFs first, then) ---
sbatch server_eval.sh 0 500 16 full
```

## Reference numbers
SWE-Master paper: 32B-SFT reaches ~61% pass@1 on SWE-bench Verified; the 4B-SFT model
is smaller, so expect lower. This harness reports the measured resolve@1.
