"""Unit tests for `active_tests.ActiveTestScheduler`.

Run from the repo root:

    /home/temp/venv/bin/python -m pytest \
        R2E-Gym/src/r2egym/agenthub/runtime/test_active_tests.py -v

These tests pin the four state-machine transitions described in
`REWARD_ACTIVE_TEST_SELECTION.md` §3.2:
  (a) ACTIVE pass -> COOLDOWN, with the alpha reward and cooldown re-check
  (b) ACTIVE fail for k consecutive ticks -> stall-expand promotes from POOL
  (c) COOLDOWN expiry: passing -> RETIRED; failing -> regression
  (d) Active starvation: if ACTIVE becomes empty, refill from POOL
"""

import pytest

from r2egym.agenthub.runtime.active_tests import (
    ActiveTestScheduler,
    SchedulerConfig,
)


def _make(tests, **cfg):
    return ActiveTestScheduler(fail_to_pass=list(tests), config=SchedulerConfig(**cfg))


def test_init_promotes_m_init_into_active():
    s = _make(["t0", "t1", "t2", "t3"], m_init=2)
    snap = s.state_snapshot()
    assert snap["active"] == ["t0", "t1"]
    assert snap["pool"] == ["t2", "t3"]
    assert snap["cooldown"] == [] and snap["retired"] == []


def test_active_pass_moves_to_cooldown_with_alpha_reward():
    s = _make(["t0", "t1"], m_init=2, cooldown_steps=3, alpha=1.0, gamma=0.0, eta=0.0)
    evt = s.tick({"t0": True, "t1": False})
    assert evt.solves == 1
    assert evt.r_alpha == pytest.approx(1.0)
    snap = s.state_snapshot()
    assert "t0" in snap["cooldown"] and "t0" not in snap["active"]
    assert "t1" in snap["active"]


def test_stall_expand_after_k_failures_promotes_from_pool():
    cfg = dict(m_init=1, k_steps=3, expand_count=2, m_max=10, gamma=0.0, eta=0.0, beta=0.2)
    s = _make(["t0", "t1", "t2", "t3"], **cfg)
    # 3 failing ticks on t0 -> one stall-expand event, 2 promotions.
    e1 = s.tick({"t0": False})
    e2 = s.tick({"t0": False})
    e3 = s.tick({"t0": False})
    assert e1.expansions == 0 and e2.expansions == 0
    assert e3.expansions == 1 and e3.promoted == 2
    assert e3.r_beta == pytest.approx(-0.2)
    snap = s.state_snapshot()
    assert snap["active"][:3] == ["t0", "t1", "t2"]
    assert snap["pool"] == ["t3"]


def test_cooldown_expiry_passing_goes_to_retired():
    s = _make(["t0"], m_init=1, cooldown_steps=2, gamma=0.0, eta=0.0)
    s.tick({"t0": True})  # tick1: t0 -> cooldown (timer=2)
    # tick2: timer 2 -> 1
    s.tick({"t0": True})
    # tick3: timer 1 -> 0 -> expire. Still green => RETIRED.
    evt = s.tick({"t0": True})
    snap = s.state_snapshot()
    assert snap["retired"] == ["t0"]
    assert snap["cooldown"] == [] and snap["active"] == []
    assert evt.regressions == 0


def test_cooldown_expiry_failing_counts_as_regression():
    s = _make(["t0"], m_init=1, cooldown_steps=1, delta=0.5, gamma=0.0, eta=0.0)
    s.tick({"t0": True})  # t0 -> cooldown (timer=1)
    # next tick: timer expires; we report it as failing -> regression
    evt = s.tick({"t0": False})
    assert evt.regressions == 1
    assert evt.r_delta == pytest.approx(-0.5)
    snap = s.state_snapshot()
    assert "t0" in snap["active"]
    assert snap["retired"] == []


def test_active_empty_refills_from_pool():
    s = _make(["t0", "t1", "t2"], m_init=1, cooldown_steps=10, gamma=0.0, eta=0.0)
    # Solve the only active test -> moves to cooldown -> ACTIVE empty -> refill.
    evt = s.tick({"t0": True})
    snap = s.state_snapshot()
    assert snap["active"], "ACTIVE was not refilled from POOL after starvation"
    # m_init=1, so one of t1/t2 should now be active.
    assert snap["active"] in (["t1"], ["t2"])


def test_dense_reward_decomposition_sums_correctly():
    s = _make(
        ["t0", "t1", "t2"],
        m_init=2, k_steps=10, cooldown_steps=10,
        alpha=1.0, beta=0.2, delta=0.5, gamma=0.01, eta=0.05,
    )
    evt = s.tick({"t0": True, "t1": False})
    expected = evt.r_alpha + evt.r_beta + evt.r_delta + evt.r_gamma + evt.r_eta
    assert evt.reward == pytest.approx(expected)
    # alpha=1 from solving t0; beta/delta=0; gamma=-0.01; eta=0.05/2=0.025
    assert evt.r_alpha == pytest.approx(1.0)
    assert evt.r_gamma == pytest.approx(-0.01)
    assert evt.r_eta == pytest.approx(0.05 / (len(s.active) + 1))


def test_m_max_caps_promotions():
    s = _make(
        ["t0", "t1", "t2", "t3", "t4"],
        m_init=1, k_steps=1, expand_count=3, m_max=2, gamma=0.0, eta=0.0,
    )
    # k_steps=1 means a fail on tick 1 immediately triggers stall-expand.
    evt = s.tick({"t0": False})
    # m_max=2 with t0 already active means at most 1 can be promoted, not 3.
    assert evt.promoted == 1
    assert len(s.active) == 2


def test_terminal_reward_uses_full_pass_map():
    s = _make(["t0", "t1", "t2", "t3"], m_init=2, w_solve=5.0, lam_token=0.1, token_ref=1000)
    r, br = s.terminal_reward(
        full_pass_map={"t0": True, "t1": True, "t2": False, "t3": True},
        tokens_used=500,
    )
    assert br["frac_solved"] == pytest.approx(0.75)
    assert br["r_solve"] == pytest.approx(5.0 * 0.75)
    assert br["r_token"] == pytest.approx(-0.1 * 0.5)
    assert r == pytest.approx(br["r_solve"] + br["r_token"])


def test_terminal_reward_falls_back_to_internal_state():
    s = _make(["t0", "t1"], m_init=2, cooldown_steps=10, w_solve=4.0, lam_token=0.0)
    # Solve t0; t1 stays failing.
    s.tick({"t0": True, "t1": False})
    r, br = s.terminal_reward(full_pass_map=None, tokens_used=0)
    # t0 is in cooldown, presumed passing -> solved={t0} -> 0.5 of 2 tests.
    assert br["frac_solved"] == pytest.approx(0.5)
    assert r == pytest.approx(4.0 * 0.5)


def test_prompt_block_well_formed_and_bounded():
    s = _make(["aaaaaaaaaa", "bbbbbbbbbb"], m_init=2, max_block_chars=200, show_bodies=False)
    block = s.prompt_block()
    assert block.startswith("<active_tests count=\"2\">")
    assert "aaaaaaaaaa" in block and "bbbbbbbbbb" in block
    assert block.endswith("</active_tests>")
    assert len(block) <= 200


def test_prompt_block_empty_when_no_active():
    s = _make([], m_init=0)
    block = s.prompt_block()
    assert block == "<active_tests count=\"0\"/>"


def test_full_episode_end_to_end_small():
    """Simulate a 20-tick episode and assert invariants hold throughout."""
    s = _make(
        ["t0", "t1", "t2", "t3", "t4"],
        m_init=2, k_steps=3, cooldown_steps=2, m_max=4,
        alpha=1.0, beta=0.2, delta=0.5, gamma=0.01, eta=0.05,
    )
    # Schedule: agent solves a different test every other tick; rest fail.
    pattern = [
        {"t0": True}, {}, {"t1": True}, {}, {"t2": True},
        {}, {"t3": True}, {}, {"t4": True}, {},
        {}, {}, {}, {}, {},
        {}, {}, {}, {}, {},
    ]
    total = 0.0
    for tick_inputs in pattern:
        evt = s.tick(tick_inputs)
        total += evt.reward
        # Bucket invariant: |active|+|cooldown|+|pool|+|retired| == N
        snap = s.state_snapshot()
        assert (
            len(snap["active"]) + len(snap["cooldown"])
            + len(snap["pool"]) + len(snap["retired"])
        ) == len(s.fail_to_pass)
        assert len(snap["active"]) <= s.config.m_max
    # Reward signal should be substantively positive given five solves.
    assert total > 0.0
    # The schedule never re-passes a test after solving it, so every cooldown
    # expiry is a regression; retirements only happen if the agent keeps a
    # solved test passing across cooldown. That's fine; assert instead that
    # at least one regression was observed (closed loop on the cooldown path).
    assert s.total_regressions >= 1, (
        f"expected at least one cooldown regression; got {s.total_regressions}"
    )


# ----------------------------------------------------------------------
# PASS_TO_PASS (regression-guard) tests
# ----------------------------------------------------------------------


def test_p2p_initial_state_all_trusted():
    s = ActiveTestScheduler(
        fail_to_pass=["f0"], pass_to_pass=["p0", "p1", "p2"],
        config=SchedulerConfig(),
    )
    assert all(s.p2p_status.values()), "all P2P should start trusted (passing)"
    assert s.total_p2p_breaks == 0
    assert s.total_p2p_recovers == 0


def test_p2p_tick_with_no_p2p_map_does_not_change_state():
    s = _make(["f0"], m_init=1, gamma=0.0, eta=0.0)
    s.pass_to_pass = ["p0", "p1"]
    s.__post_init__()  # rebuild with the new pass_to_pass
    snap_before = dict(s.p2p_status)
    evt = s.tick({"f0": False})  # no p2p_pass_map
    assert s.p2p_status == snap_before
    assert evt.p2p_newly_broken == 0
    assert evt.p2p_newly_recovered == 0
    assert evt.r_p2p_break == 0.0
    assert evt.r_p2p_recover == 0.0


def test_p2p_break_applies_dense_penalty():
    s = ActiveTestScheduler(
        fail_to_pass=["f0"], pass_to_pass=["p0", "p1"],
        config=SchedulerConfig(
            m_init=1, gamma=0.0, eta=0.0, delta_p2p_break=0.5, delta_p2p_recover=0.25
        ),
    )
    evt = s.tick({"f0": False}, p2p_pass_map={"p0": False, "p1": True})
    assert evt.p2p_newly_broken == 1
    assert evt.p2p_newly_recovered == 0
    assert evt.r_p2p_break == pytest.approx(-0.5)
    assert evt.n_p2p_broken == 1
    assert s.p2p_status == {"p0": False, "p1": True}


def test_p2p_recover_applies_dense_bonus():
    s = ActiveTestScheduler(
        fail_to_pass=["f0"], pass_to_pass=["p0"],
        config=SchedulerConfig(
            m_init=1, gamma=0.0, eta=0.0, delta_p2p_break=0.5, delta_p2p_recover=0.25
        ),
    )
    # Break, then recover.
    e1 = s.tick({"f0": False}, p2p_pass_map={"p0": False})
    assert e1.p2p_newly_broken == 1
    assert s.p2p_status == {"p0": False}
    e2 = s.tick({"f0": False}, p2p_pass_map={"p0": True})
    assert e2.p2p_newly_recovered == 1
    assert e2.r_p2p_recover == pytest.approx(0.25)
    assert s.p2p_status == {"p0": True}


def test_terminal_reward_with_p2p_break_subtracts_w_regress():
    s = ActiveTestScheduler(
        fail_to_pass=["f0", "f1"], pass_to_pass=["p0", "p1", "p2", "p3"],
        config=SchedulerConfig(w_solve=10.0, w_regress=5.0, lam_token=0.0),
    )
    r, br = s.terminal_reward(
        full_pass_map={"f0": True, "f1": True},
        p2p_pass_map={"p0": True, "p1": False, "p2": False, "p3": True},
        tokens_used=0,
    )
    assert br["frac_solved"] == pytest.approx(1.0)
    assert br["r_solve"] == pytest.approx(10.0)
    assert br["frac_p2p_broken"] == pytest.approx(0.5)  # 2 of 4 broken
    assert br["r_regress"] == pytest.approx(-2.5)
    assert r == pytest.approx(10.0 - 2.5)


def test_terminal_reward_no_p2p_map_falls_back_to_internal_state():
    s = ActiveTestScheduler(
        fail_to_pass=["f0"], pass_to_pass=["p0", "p1"],
        config=SchedulerConfig(w_solve=10.0, w_regress=5.0, lam_token=0.0),
    )
    s.tick({"f0": False}, p2p_pass_map={"p0": False, "p1": True})  # break p0
    r, br = s.terminal_reward(full_pass_map={"f0": False}, tokens_used=0)
    assert br["frac_p2p_broken"] == pytest.approx(0.5)
    assert br["r_regress"] == pytest.approx(-2.5)


def test_terminal_reward_no_p2p_at_all_is_unaffected():
    s = ActiveTestScheduler(
        fail_to_pass=["f0", "f1"], pass_to_pass=[],
        config=SchedulerConfig(w_solve=10.0, w_regress=5.0, lam_token=0.0),
    )
    r, br = s.terminal_reward(
        full_pass_map={"f0": True, "f1": True}, tokens_used=0,
    )
    assert br["frac_p2p_broken"] == 0.0
    assert br["r_regress"] == 0.0
    assert r == pytest.approx(10.0)


def test_p2p_total_counters_accumulate_across_ticks():
    s = ActiveTestScheduler(
        fail_to_pass=["f0"], pass_to_pass=["p0", "p1"],
        config=SchedulerConfig(m_init=1, gamma=0.0, eta=0.0),
    )
    s.tick({"f0": False}, p2p_pass_map={"p0": False, "p1": True})
    s.tick({"f0": False}, p2p_pass_map={"p0": False, "p1": False})
    s.tick({"f0": False}, p2p_pass_map={"p0": True, "p1": True})
    # tick1: p0 broken (1 break)
    # tick2: p1 broken (1 break)
    # tick3: p0+p1 recovered (2 recovers)
    assert s.total_p2p_breaks == 2
    assert s.total_p2p_recovers == 2


def test_token_penalty_is_capped_against_pathological_output():
    """A trajectory with absurd token usage (e.g. cat-ing a huge file) must not
    explode the reward. The token penalty is clamped at token_penalty_cap.
    Regression test for the −989 r_token / grad_norm=5.4 spike observed in the
    paper150 run at step 8.
    """
    s = ActiveTestScheduler(
        fail_to_pass=["t0"],
        config=SchedulerConfig(
            w_solve=10.0, lam_token=0.1, token_ref=32000.0,
            token_penalty_cap=3.0,
        ),
    )
    # 317 million "tokens" (chars) — the pathological value we actually hit.
    r, br = s.terminal_reward(full_pass_map={"t0": False}, tokens_used=316_000_000)
    # Without the cap this would be ~ -987. With cap=3.0: -0.1 * 3.0 = -0.3.
    assert br["r_token"] == pytest.approx(-0.3)
    assert r >= -0.3 - 1e-6, f"token penalty not capped: r={r}"


def test_token_penalty_normal_case_unaffected_by_cap():
    s = ActiveTestScheduler(
        fail_to_pass=["t0"],
        config=SchedulerConfig(w_solve=10.0, lam_token=0.1, token_ref=32000.0,
                               token_penalty_cap=3.0),
    )
    # 16k tokens → ratio 0.5 → penalty -0.05 (well under the cap)
    r, br = s.terminal_reward(full_pass_map={"t0": True}, tokens_used=16000)
    assert br["r_token"] == pytest.approx(-0.05)
    assert br["r_solve"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# B1 / R12 — online refinement annotation (prompt-only; reward must be UNCHANGED)
# ---------------------------------------------------------------------------


def _refine_stub(test_id):
    """A pure decomposition callback: split a test into two minimal sub-tests."""
    return [
        (f"{test_id}__sub1", "def test_sub1():\n    assert target_behavior() == 1"),
        (f"{test_id}__sub2", "def test_sub2():\n    assert edge_case() is None"),
    ]


def test_refinement_is_off_by_default():
    s = _make(["t0", "t1", "t2"], m_init=1, k_steps=3)
    assert s.config.refine_enabled is False
    s.refine_fn = _refine_stub  # even with a fn set, default-off => no annotation
    for _ in range(6):
        s.tick({})  # everything failing
    assert s.refined == {}
    assert s.total_refinements == 0


def test_refine_after_sentinel_resolves_to_k_steps():
    assert SchedulerConfig(k_steps=4, refine_after=-1).refine_after == 4
    assert SchedulerConfig(k_steps=4, refine_after=2).refine_after == 2


def test_refine_annotation_appears_in_prompt_after_stall():
    s = _make(["t0", "t1", "t2"], m_init=1, k_steps=5,
              refine_enabled=True, refine_after=2)
    s.refine_fn = _refine_stub
    s.tick({"t0": False})   # stall counter t0 -> 1
    assert "t0" not in s.refined
    s.tick({"t0": False})   # stall counter t0 -> 2 == refine_after -> annotate
    assert "t0" in s.refined
    assert s.total_refinements == 1
    block = s.prompt_block()
    assert 'refined="true"' in block
    assert "t0__sub1" in block and "t0__sub2" in block


def test_reward_stream_identical_refine_on_vs_off():
    """The HARD CONSTRAINT: per-tick reward + terminal reward are bit-identical
    with refinement on vs off, given the same pass_map stream."""
    common = dict(m_init=2, k_steps=3, cooldown_steps=2, m_max=4, expand_count=1)
    s_off = _make(["t0", "t1", "t2", "t3"], refine_enabled=False, **common)
    s_on = _make(["t0", "t1", "t2", "t3"], refine_enabled=True, refine_after=2, **common)
    s_on.refine_fn = _refine_stub

    stream = [
        {"t0": False, "t1": False},   # stall both
        {"t0": False, "t1": False},   # refine_after hit on 'on' (annotate)
        {"t0": True, "t1": False},    # t0 solved -> cooldown
        {"t1": False},                # t1 stalls -> expand
        {"t0": False},                # cooldown re-check / regress path
        {"t0": False, "t1": True, "t2": False, "t3": False},
        {},                           # all failing
        {"t2": True, "t3": True},
    ]
    r_off = [s_off.tick(dict(pm)).reward for pm in stream]
    r_on = [s_on.tick(dict(pm)).reward for pm in stream]
    assert r_off == r_on, f"reward stream diverged:\noff={r_off}\non ={r_on}"

    # The 'on' scheduler really did refine (so the match is meaningful, not vacuous).
    assert s_on.total_refinements > 0
    assert s_off.total_refinements == 0

    # Terminal reward must also be bit-identical.
    term_pm = {"t0": True, "t1": False, "t2": True, "t3": False}
    ro, bro = s_off.terminal_reward(full_pass_map=dict(term_pm), tokens_used=12345)
    rn, brn = s_on.terminal_reward(full_pass_map=dict(term_pm), tokens_used=12345)
    assert ro == rn and bro == brn


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
