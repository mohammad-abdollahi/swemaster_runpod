# Resume the integrated online test-to-OpenAI refinement run

**Paused by the user at global_step_5** (5 of 10 PPO steps) on 2026-06-25. Branch
`test-refinement-test-to-openai`. Step 6 was mid-rollout when stopped and was NOT
checkpointed — last fully-saved state is `global_step_5`.

## What this run is
Single ATS-RL run with refinement INTEGRATED into the loop: on stall, gpt-5.4-mini
rewrites/splits the failing test live and the refined test REPLACES the scored test
(drives `frac_solved`). Dataset `swerebench_verified_live` (64 train / 8 val), batch 16,
SWE-Master-4B-SFT base, 10 steps, refine_after=2, w_solve=5.0 (original ATS-RL reward).

## Reward trend so far (LEARNING — net rising)
step1=+0.016 → step2=−0.219 → step3=+0.146 → step4=+0.360 → step5=+0.242
(after the step-2 dip it climbed into a clearly positive regime; ~16–28% of episodes
solve their refined test set; 0 crashes). Costs on gpt-5.4-mini (the first 85 of 326
cached refinements were gpt-5.4 before the cost switch; all new ones are mini).

## Saved
| Artifact | Path | For |
|---|---|---|
| HF model (5 steps) | `/home/temp/models/SWE-Master-4B-TestToOpenAI-online` (8.3G) | usable model / HF-weight continuation |
| verl ckpt + **optimizer** | `…/training_outputs/online_refine/online/global_step_5/` (63G) | **true resume** |
| prior ckpt | `…/global_step_4/` | safety |
| refinement cache | `block_b_refinement/cache/online_refine_swerebench_verified_live.jsonl` (326) | reused on resume (committed to git) |
| dataset | `data_examples/rl_data/swerebench_verified_live/` (64/8) | committed |

> The 63G verl ckpts are on disk ONLY (too big for git). If the disk is wiped, only the
> 8.3G HF model survives → use Option B.

## How to resume

### Option A — TRUE resume from global_step_5 (keeps optimizer/LR; finishes steps 6–10)
verl auto-resumes from the highest `global_step_*` in the output dir. Just relaunch the
orchestrator unchanged:
```bash
cd /home/temp/swemaster_runpod
setsid nohup bash online_refine_orchestrator.sh >/dev/null 2>&1 &
```
Requires `…/online_refine/online/global_step_5/` to still exist. Refinement cache is
reused (no re-pay); new stalls call gpt-5.4-mini.

### Option B — HF-weight continuation from the exported model (optimizer reset; fresh run)
Use if the verl ckpt is gone but the HF model survives. Point BASE_MODEL at the export
and use a fresh output dir:
```bash
BASE_MODEL=/home/temp/models/SWE-Master-4B-TestToOpenAI-online \
DATASET=swerebench_verified_live STEPS=10 BATCH=16 \
setsid nohup bash online_refine_orchestrator.sh >/dev/null 2>&1 &
# (online_refine_orchestrator.sh hardcodes SFT base; for B, edit SFT= to the export path,
#  or run run_test_to_openai_arm.sh directly with BASE_MODEL + RLLM_ACTIVE_TESTS_ONLINE_REFINE=1)
```

## On every restart also
- Relaunch the miner watchdog: `setsid nohup bash miner_killer.sh >/dev/null 2>&1 &`
- If `cudaa` miners or the money4bands stack (ali_*/honeygain/etc.) respawn, re-kill per
  `[[feedback-kill-miners]]` memory (kill containers + `rm -rf ~/.local/share/money4band`
  + rmi images by name — NEVER `docker image prune -af`).
- Orphaned `sweb.eval` containers accumulate ~64 per trainer kill-9 restart; clean ones
  >40min old by ID (active rollout is <25min). See the cleanup recipe in MEMORY.
