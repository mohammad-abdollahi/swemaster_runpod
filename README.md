<h1 align="center">Adaptive Test Selection and Refinement for Reinforcement Learning in Program Repair and Unit Test Generation</h1>

<p align="center"><i>Anonymous artifact for double-blind review.</i></p>

This repository contains the code for the repository-level program repair experiments in the paper. It trains a 4B-parameter software engineering agent with reinforcement learning, replacing the usual binary pass/fail reward with an adaptive verification signal:

1. **Dense test-level reward.** Partial credit for the fraction of target (fail-to-pass) tests a patch resolves, minus a penalty for broken regression (pass-to-pass) tests and a capped penalty for token consumption.
2. **Policy-conditioned test selection.** Each training instance keeps its target tests in *active*, *pool*, and *cooldown* states. Passed tests cool down and are re-checked; a stalled active set is expanded from the pool.
3. **Dynamic test refinement.** Tests the policy repeatedly fails are rewritten or decomposed by an LLM-based refiner. A refined test is accepted only if it fails on the buggy repository and passes once the reference patch is applied.

The implementation builds on the public SWE-Master RL pipeline (reference [3] in the paper) and its underlying open-source frameworks. The model initialization, optimizer, agent scaffold, tools, and verifier execution pipeline are inherited from that baseline; the changes in this repository are on the reward and verification side.

## What this repository covers

| Paper component | Where it lives |
|---|---|
| Dense reward (Eq. 2–3) | `ActiveTestScheduler.terminal_reward` in `R2E-Gym/src/r2egym/agenthub/runtime/active_tests.py` |
| Test selection state machine (active / pool / cooldown, stall expansion) | `ActiveTestScheduler.tick` in the same file; unit tests in `test_active_tests.py` next to it |
| Reward wiring into the RL environment | `DeepSWE_RL/rllm/rllm/environments/swe/swe_arsenal.py` |
| Dynamic refinement with an agentic refiner (repository read and execute access, trajectory diversity filter, fail-on-buggy / pass-on-patched validation) | `R2E-Gym/src/r2egym/agenthub/runtime/agentic_refine.py` |
| Dynamic refinement with a single-call API refiner | `R2E-Gym/src/r2egym/agenthub/runtime/online_refine.py` |
| Offline refinement (predefined tests) | `test_to_openai_refiner.py`, `build_refined_dataset.py`, `build_decomposition_cache.py`, shared helpers in `R2E-Gym/src/r2egym/agenthub/runtime/test_refiner.py` |
| Training launchers | `DeepSWE_RL/rllm/examples/swe/train_active_tests_swe_master_4b_h200.sh` (base launcher), `run_agentic_arm.sh`, `run_test_to_openai_arm.sh`, `agentic_orchestrator.sh` |
| Log analysis and plotting | `analyze_*.py`, `dump_metrics_csv.py`, `plot_*.py`, `simulate_binary_baseline.py` |

Not included in this snapshot:

- The run-level refinement triggers (reward band and reward trend). In this code, refinement is triggered per test, once a test has stayed unsolved for `RLLM_ACTIVE_TESTS_REFINE_AFTER` scheduler ticks.
- The scripts that compute mutation score and line/branch coverage for original versus refined tests.
- The CodeContests unit-test-generation experiments built on CURE.
- The eight-GPU launch configuration used for the paper runs. The launcher here is the single-GPU version of the same pipeline (`trainer.n_gpus_per_node=1`); the GPU count, tensor-parallel size, and memory settings in it need to be raised for a multi-GPU node.
- The full 800-instance training parquet. `data_examples/rl_data/` holds smaller subsets in the same format, and the build scripts below reproduce the format from public sources.

## Repository layout

```
DeepSWE_RL/rllm/          RL trainer (rLLM + verl), SWE environment, launch scripts
R2E-Gym/                  Agent scaffold, Docker runtime, test scheduler, refiners
OpenRLHF_SFT/             Supervised fine-tuning scripts inherited from the baseline (not used by the RL experiments)
data_preparation/         Dataset download, difficulty filtering, test-spec generation
data_examples/            Example RL parquets, inference data, and trajectories
block_b_refinement/       Refinement caches
build_*.py                Dataset builders (SWE-rebench / SWE-Gym subsets, refined datasets)
run_*_arm.sh              Training launchers for the refinement configurations
```

## Setup

Requirements: Linux, Python 3.11, at least one 80 GB GPU (the paper runs used eight H100s), and a Docker daemon the training user can access (each rollout runs in a per-instance container).

```bash
# from the repository root
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.11 ../venv && source ../venv/bin/activate

uv pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1 torchvision torchaudio

# verl (the fork the rLLM trainer expects), cloned inside the trainer directory
git clone --depth 1 https://github.com/agentica-project/verl.git DeepSWE_RL/rllm/verl
uv pip install -e DeepSWE_RL/rllm/verl

uv pip install -e R2E-Gym
uv pip install ./swebench_fork_swegym-2.0.13-py3-none-any.whl \
               ./swebench_fork_swerebench-4.0.3-py3-none-any.whl \
               ./swesmith-0.0.7-py3-none-any.whl
uv pip install -e DeepSWE_RL/rllm --no-deps

uv pip install vllm==0.10.2
uv pip install wheel packaging && uv pip install --no-build-isolation flash-attn==2.8.3
uv pip install 'setuptools<81' 'transformers>=4.45,<5' 'huggingface_hub<1' tensorboard
```

`build_env.sh` runs the same steps. `RL_ENV.txt` lists the full package set of the reference environment.

The launch and helper scripts were written for a machine with the repository at `/home/temp/swemaster_runpod` and the virtual environment at `/home/temp/venv`, and several of them hard-code those paths. Either place (or symlink) the checkout and environment there, or replace the prefix in the scripts you use.

Check the scheduler independently of the GPU stack:

```bash
python -m pytest R2E-Gym/src/r2egym/agenthub/runtime/test_active_tests.py -v
```

## Data

Training uses 800 issue-resolution instances from 706 repositories: 700 from SWE-rebench and 100 from SWE-Gym. Each instance has at least one fail-to-pass test, one pass-to-pass test, a reference patch, and a runnable container image. SWE-rebench instances are filtered on issue specification and test validity, with a 25% / 50% / 25% easy / medium / hard split, reference patches touching at most three files, and issue descriptions between 200 and 10,000 characters. Repositories that appear in SWE-bench Verified are excluded.

```bash
# download the source datasets
bash ./data_preparation/download_swe_datasets.sh

# optional: attach difficulty scores and filter on them
python ./data_preparation/difficulty_score_add.py
python ./data_preparation/difficulty_score_filter.py

# embed test specs so the runtime needs no network access during training
python ./data_preparation/make_test_spec.py --base_dir /path/to/cache --data_file_path /path/to/data.json
python ./data_preparation/make_test_spec_for_rl.py
```

`build_comprehensive_swerebench.py` builds a training parquet directly from the public SWE-rebench release, keeping only instances whose images exist on Docker Hub. It has absolute paths near the top that need to point at your checkout. Pre-pull the container images before training (`prepull_parquet_images.sh`) so rollouts do not stall on downloads.

The trainer reads `train_00000_verl.parquet` / `test_00000_verl.parquet` pairs; see `data_examples/rl_data/` for examples.

## Training

All configurations go through one base launcher, `DeepSWE_RL/rllm/examples/swe/train_active_tests_swe_master_4b_h200.sh`, which reads its settings from environment variables. Set the common ones once:

```bash
export MODEL_PATH=/path/to/SWE-Master-4B-SFT          # initial supervised checkpoint
export BASE_MODEL=$MODEL_PATH                         # the run_*_arm.sh wrappers read this name
export TRAIN_PARQUET=/path/to/train_00000_verl.parquet
export VAL_PARQUET=/path/to/test_00000_verl.parquet

# paper setting: 16 instances x 4 trajectories per step, 200 optimizer steps, 150-turn episodes
export TRAIN_BATCH_SIZE=16 PPO_MINI_BATCH_SIZE=16 TOTAL_TRAINING_STEPS=200 TOTAL_EPOCHS=40
export AGENT_MAX_STEPS=150 TRAJECTORY_TIMEOUT=5400 MAX_RESPONSE=24576
```

Then launch one of the configurations:

```bash
LAUNCH=DeepSWE_RL/rllm/examples/swe/train_active_tests_swe_master_4b_h200.sh

# 1) Binary-reward RL baseline
EXPERIMENT_NAME=binary RLLM_ACTIVE_TESTS_ENABLED=0 bash $LAUNCH

# 2) Dense reward + test selection, no refinement
EXPERIMENT_NAME=selection RLLM_ACTIVE_TESTS_ENABLED=1 bash $LAUNCH

# 3) Predefined tests: refine offline, then train with (2) on the refined parquet.
#    The refiner reads failed trajectories dumped by an earlier run launched with
#    RLLM_DUMP_TRAJ_DIR=/path/to/dumps, and an API key from API_KEY_OPENAI (env or .env).
python test_to_openai_refiner.py --dump-dir /path/to/dumps --train-parquet "$TRAIN_PARQUET" \
    --out block_b_refinement/cache/refinements_test_to_openai.jsonl
python build_refined_dataset.py --src /path/to/original_dataset_dir \
    --cache block_b_refinement/cache/refinements_test_to_openai.jsonl --out /path/to/refined_dataset_dir
EXPERIMENT_NAME=predefined RLLM_ACTIVE_TESTS_ENABLED=1 \
  TRAIN_PARQUET=/path/to/refined_dataset_dir/train_00000_verl.parquet bash $LAUNCH

# 4) Dynamic refinement with the agentic refiner: validated refined tests replace the stalled test
EXPERIMENT_NAME=dynamic bash run_agentic_arm.sh

# 4b) Dynamic refinement with a single API call per stalled test
EXPERIMENT_NAME=dynamic_api RLLM_ACTIVE_TESTS_REFINE=1 RLLM_ACTIVE_TESTS_ONLINE_REFINE=1 \
  bash run_test_to_openai_arm.sh
```

Notes:

- `run_agentic_arm.sh` drives a headless coding-agent CLI as the refiner; point `RLLM_AGENTIC_CLAUDE_BIN` at the binary. `RLLM_AGENTIC_REFINE_MODEL` selects the refiner model.
- With test selection enabled, the active test identifiers are appended to the task instruction (`swe_arsenal.py`).
- `prepull_parquet_images.sh <parquet>` pulls every container image a parquet references.
- `agentic_orchestrator.sh` wraps a long run with relaunch-on-failure and a final export to a Hugging Face model directory.
- Checkpoints are written under `/home/temp/training_outputs/<PROJECT_NAME>/<EXPERIMENT_NAME>/`.

### Reward and scheduler settings

All settings are environment variables read in `swe_arsenal.py`. Paper symbols are from Eq. 2–3 and Section III-D.

| Variable | Symbol | Default | Meaning |
|---|---|---|---|
| `RLLM_ACTIVE_TESTS_ENABLED` | – | 1 | 0 restores the binary reward |
| `RLLM_ACTIVE_TESTS_W_SOLVE` | w_s | 5.0 | Weight on the fraction of target tests passed |
| `RLLM_ACTIVE_TESTS_W_REGRESS` | w_r | w_s / 2 | Weight on the fraction of regression tests broken |
| `RLLM_ACTIVE_TESTS_LAM_TOKEN` | λ | 0.1 | Token penalty weight |
| `RLLM_ACTIVE_TESTS_TOKEN_REF` | T_ref | 32000 | Token normalization constant |
| `RLLM_ACTIVE_TESTS_TOKEN_PENALTY_CAP` | c | 3.0 | Upper bound on the normalized token term |
| `RLLM_ACTIVE_TESTS_M_INIT` / `_M_MAX` | m_init / m_max | 2 / 6 | Bounds on the active set size |
| `RLLM_ACTIVE_TESTS_K` | k | 5 | Stall window before expansion |
| `RLLM_ACTIVE_TESTS_EXPAND` | – | 2 | Tests promoted from the pool per stall expansion |
| `RLLM_ACTIVE_TESTS_COOLDOWN` | – | 10 | Cooldown length before a passed test is re-checked |
| `RLLM_ACTIVE_TESTS_MID_EPISODE` | – | 0 | 1 runs the tests every k agent steps and adds per-tick shaping terms (`ALPHA`, `BETA`, `DELTA`, `GAMMA`, `ETA`) |
| `RLLM_ACTIVE_TESTS_REFINE_AFTER` | – | 2 (refinement wrappers) | Ticks a test must stay unsolved before it is refined |
| `RLLM_ACTIVE_TESTS_REFINE_MAX` | – | 4 | Maximum tests produced when one test is decomposed |
| `RLLM_AGENTIC_MAX_TRAJ` / `RLLM_AGENTIC_TRAJ_DIVERSITY` | – | 3 / 0.30 | Number of failed trajectories given to the refiner, and the minimum dissimilarity for one to be included |

### Monitoring

```bash
# the launcher logs to stdout; redirect it to a file, e.g. `bash $LAUNCH > train.log 2>&1`
grep -aoE 'critic/rewards/mean:-?[0-9.]+' train.log | tail   # reward per step
python analyze_active_tests.py train.log                     # per-step reward breakdown
python dump_metrics_csv.py train.log metrics.csv             # per-step CSV for plotting
```

## Evaluation on SWE-bench Verified

Evaluation uses the original SWE-bench Verified tests; refined tests are never used here.

```bash
# export the final checkpoint to a Hugging Face model directory
bash export_to_hf.sh /path/to/experiment_dir /path/to/exported_model

# serve it (set MODEL_NAME inside the script)
bash ./R2E-Gym/rollout_trained_model/vllm_server_launch.sh

# run the agent on the 500 instances (set the dataset path and server address inside the script)
bash ./R2E-Gym/rollout_trained_model/qwen25_coder_15k_lsp_with_gerenal_04.sh
```

Resolve rate and token usage are computed from the trajectory files written to `--traj_dir`.

## Acknowledgements

This code builds on several open-source projects: [SWE-Master](https://arxiv.org/abs/2602.03411), [R2E-Gym](https://github.com/R2E-Gym/R2E-Gym), [rLLM / DeepSWE](https://github.com/agentica-project/rllm), [verl](https://github.com/verl-project/verl), and [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF). Training data comes from SWE-rebench and SWE-Gym, and evaluation uses SWE-bench Verified.

## License

Released under the [MIT License](LICENSE).
