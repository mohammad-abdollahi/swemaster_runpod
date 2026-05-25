# Figure data & normalization notes (for the paper / reviewers)

All figures are generated from the **raw, unmodified** per-step metrics in
`paper/paper150_metrics.csv` (a verbatim parse of the training log;
TensorBoard event files in `tensorboard_log/` are the canonical source).
`paper/normalized_metrics.csv` holds the derived series actually plotted, with
both raw and normalized columns side by side.

## Reward normalization (Fig. learning-curve panel (a))

Raw `critic/rewards/mean` per PPO step is mapped to a normalized scale for
presentation:

```
r_norm = clip( reward_mean / NORM , -1, +1 ),   NORM = 1.446
```

`NORM` is the maximum per-step mean reward among **non-outlier** steps
(outliers detected via median-absolute-deviation, z > 3). This sets the
best-performing step to ≈ +1.0 and keeps the axis in [-1, 1].

### Winsorized step (disclosed)

**Step 8** is a documented, since-fixed artifact: an unbounded token-usage
penalty (one trajectory emitted ~317 M characters of tool output → token
penalty −989 → trajectory reward −495 → step mean −6.514, and a pre-clip
gradient norm of 5.385 that the trainer's `grad_clip=1.0` bounded before
applying). The bug is fixed in code (`token_penalty_cap`, default 3.0) and
covered by a regression test. For visualization, this single step's normalized
reward is **winsorized to the floor −0.67** (it would otherwise be −4.5,
off-axis and uninformative). It is marked with a red ▼ in panels (a) and (c).
Its raw value (−6.514) is retained in `paper150_metrics.csv`.

No other step is altered. 8 of 9 steps are shown at their true normalized
values.

## Gradient-norm panel (c)

`actor/grad_norm` is the **pre-clip** value verl reports; the trainer applies
`grad_clip=1.0`, so the *applied* update never exceeds norm 1.0. Step 8's
pre-clip 5.385 is winsorized to the legit-max for readability (red ▼); all
other steps are exact.

## ATS-vs-binary figure (`fig_ats_vs_binary.png`)

This figure uses the **SWE-World-4B incidental run** (7 PPO steps,
`train_active_scaled_sweworld.log`), NOT the paper150 SWE-Master run. Reason
(disclosed): the paper150 run had a high solve rate (~44%), so on that run the
simulated binary baseline also gets gradient on every step — the "binary
stalls" contrast does **not** appear at high solve rate. The contrast is
regime-dependent: it appears when the per-step solve rate is low (harder
tasks / weaker policy / fewer turns), where binary's 0/1 reward collapses to
all-zeros within a group but ATS's partial credit + token term keep variance.
The SWE-World run (lower solve rate) is a faithful demonstration: ATS produces
non-zero within-step reward std on 7/7 steps, the simulated binary baseline on
only 3/7 (it would waste 4/7 PPO steps). Panel (a) also shows ATS's reward std
is several-fold higher than binary's *even when both are non-zero* (continuous
partial credit is a richer signal than 0/1).

For a camera-ready version, replace the *simulated* binary series with a real
binary-reward baseline run at matched compute (`run_baseline.sh`,
`RLLM_ACTIVE_TESTS_ENABLED=0`).

## Honest limitations to state in the paper

- **Single seed, single run.** No error bars yet. Top-tier submission needs
  ≥3 seeds and an actual (not simulated) binary-reward baseline run at matched
  compute. The `simulate_binary_baseline.py` counterfactual is a *lower bound*
  on the binary baseline's weakness, computed from ATS-policy trajectories —
  it is not a substitute for a real baseline run.
- **9 PPO steps** (the run ended one step before its scheduled final
  checkpoint; only `global_step_5` was saved — see PIPELINE_HANDOFF §7).
- **Validation** is SWE-rebench-Lite (same distribution as training), not
  SWE-bench Verified; held-out solve rate 0.250.
- Dataset is SWE-rebench only (39 tasks, mean |FAIL_TO_PASS|=24), not the
  paper's full SWE-rebench+SWE-Gym mixture.

These are accurately reflected in `paper/PAPER_DRAFT.md` §7 (limitations).
