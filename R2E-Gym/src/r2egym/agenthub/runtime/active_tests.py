"""Active-Test-Selection scheduler for RL.

State machine over `FAIL_TO_PASS` for one SWE instance. The scheduler is asked
for an update every time the runtime collects a fresh pass/fail status map
(typically once every k agent steps). It returns a dense step reward, a
prompt-block string describing the currently-active tests, and event counts
for telemetry.

Design rationale and the original user sketch are in
``REWARD_ACTIVE_TEST_SELECTION.md`` at the repo root. Brief recap of the four
buckets each FAIL_TO_PASS test ``t`` moves through:

  ACTIVE    -- shown in the prompt; included in the dense reward bookkeeping.
  POOL      -- not yet shown; promoted to ACTIVE on stall-expand.
  COOLDOWN  -- recently passed; masked from the prompt for ``cooldown`` steps,
               then re-checked. Survives re-check -> RETIRED. Regresses -> ACTIVE.
  RETIRED   -- passed re-verification; not shown; contributes to terminal reward.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class SchedulerConfig:
    """All knobs the scheduler exposes. See `REWARD_ACTIVE_TEST_SELECTION.md` §3."""

    # State-machine timing
    k_steps: int = 5             # eval cadence; stall counter wraps at this
    cooldown_steps: int = 10     # how long a passed test stays masked
    m_init: int = 2              # initial size of ACTIVE
    m_max: int = 6               # hard cap on |ACTIVE|
    expand_count: int = 2        # tests promoted from POOL on stall-expand

    # Dense reward weights (F2P)
    alpha: float = 1.0           # per newly-solved test (the strongest signal)
    beta: float = 0.2            # per stall-expansion (cost of getting stuck)
    delta: float = 0.5           # per regression on cooldown re-check
    gamma: float = 0.01          # per scheduler tick (per eval cadence)
    eta: float = 0.05            # weight on the 1/(|ACTIVE|+1) shaping term

    # PASS_TO_PASS (regression-guard) handling. See §3.7 of the design doc.
    # P2P tests start "trusted" (assumed passing). Each scheduler tick may
    # re-verify them (mid-episode pytest path); regressions accumulate.
    # Terminal reward subtracts `w_regress * frac_p2p_broken`.
    w_regress: float = 2.5       # default = w_solve / 2 (lighter than F2P credit)
    delta_p2p_break: float = 0.5 # per newly-broken P2P this tick (dense)
    delta_p2p_recover: float = 0.25  # per P2P recovered from broken (dense)

    # Terminal reward weights (applied separately in the env wrapper)
    w_solve: float = 5.0         # multiplier on (|RETIRED ∪ COOLDOWN_passing|)/N
    lam_token: float = 0.1       # multiplier on tokens_used/token_ref
    token_ref: float = 32000.0   # token-penalty reference
    # Hard cap on the token-penalty magnitude (in units of tokens/token_ref).
    # Without this, a single trajectory with a pathological bash output (e.g.
    # cat-ing a huge file → hundreds of MB of observation text) drives
    # tokens_used to absurd values, exploding r_token (we observed −989 →
    # a −495 reward → grad_norm spike of 5.4). Cap so the token penalty can
    # never dominate the solve reward. Default 3.0 ⇒ max penalty = 3*lam_token.
    token_penalty_cap: float = 3.0

    # Prompt formatting
    max_block_chars: int = 4000  # hard cap on the injected prompt-block size
    show_bodies: bool = False    # if True, scheduler caller can attach test bodies

    # Online refinement (B1/R12) — PROMPT-ONLY, never touches the reward.
    # When a FAIL_TO_PASS test stays unsolved for `refine_after` stall ticks while
    # ACTIVE, the scheduler annotates it with a backtranslated decomposition (a list
    # of smaller sub-tests) supplied by an injected `refine_fn`. The decomposition is
    # rendered into the prompt block as guidance for that test ONLY — it is never
    # scored and never enters tick()/terminal_reward(). Default off => exact control.
    refine_enabled: bool = False
    refine_after: int = -1          # stall-tick threshold; <0 -> resolved to k_steps
    refine_max_subtests: int = 4    # cap on sub-tests rendered per refined test

    def __post_init__(self) -> None:
        if self.k_steps <= 0:
            raise ValueError(f"k_steps must be > 0 (got {self.k_steps})")
        if self.m_init < 0:
            raise ValueError(f"m_init must be >= 0 (got {self.m_init})")
        if self.m_max < self.m_init:
            raise ValueError(
                f"m_max ({self.m_max}) must be >= m_init ({self.m_init})"
            )
        if self.cooldown_steps < 0:
            raise ValueError(f"cooldown_steps must be >= 0 (got {self.cooldown_steps})")
        # Resolve the refinement sentinel: <0 means "trigger at the stall point".
        if self.refine_after < 0:
            self.refine_after = self.k_steps
        if self.refine_max_subtests < 1:
            raise ValueError(
                f"refine_max_subtests must be >= 1 (got {self.refine_max_subtests})"
            )


@dataclass
class SchedulerEvent:
    """Per-tick result returned to the env."""

    reward: float
    n_active: int
    n_cooldown: int
    n_retired: int
    n_pool: int
    # diagnostics — F2P
    solves: int = 0          # tests that turned green this tick
    expansions: int = 0      # number of stall-expand events
    regressions: int = 0     # cooldown re-checks that regressed
    promoted: int = 0        # individual tests pulled from POOL to ACTIVE
    refinements: int = 0     # stalled tests annotated with a decomposition this tick
                             # (telemetry ONLY — never part of `reward`)
    # diagnostics — P2P
    p2p_newly_broken: int = 0    # P2P tests that flipped trusted → broken
    p2p_newly_recovered: int = 0 # P2P tests that flipped broken → trusted
    n_p2p_broken: int = 0        # cumulative count of broken P2P tests
    n_p2p_total: int = 0         # total |PASS_TO_PASS|
    # shape components (for logging)
    r_alpha: float = 0.0
    r_beta: float = 0.0
    r_delta: float = 0.0
    r_gamma: float = 0.0
    r_eta: float = 0.0
    r_p2p_break: float = 0.0   # dense P2P breakage penalty (negative)
    r_p2p_recover: float = 0.0 # dense P2P recovery bonus (positive)


@dataclass
class ActiveTestScheduler:
    """Manage the four-bucket state for one task across a rollout.

    Usage::

        sched = ActiveTestScheduler(fail_to_pass=[...], config=SchedulerConfig())
        # every k_steps:
        evt = sched.tick(pass_map={test_id: bool, ...})
        # prompt injection:
        block = sched.prompt_block(test_bodies=optional_dict)
        # terminal:
        r_final = sched.terminal_reward(final_pass_map, tokens_used)
    """

    fail_to_pass: List[str]
    config: SchedulerConfig = field(default_factory=SchedulerConfig)
    pass_to_pass: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.active: "OrderedDict[str, int]" = OrderedDict()  # test -> stall counter
        self.pool: List[str] = list(self.fail_to_pass)
        self.cooldown: "OrderedDict[str, int]" = OrderedDict()  # test -> steps remaining
        self.retired: List[str] = []
        # Promote the initial ACTIVE window.
        self._promote_from_pool(self.config.m_init)
        # P2P state: each P2P test is True (trusted, presumed passing) or False
        # (broken — last observed failing). Initial state: all trusted, since
        # the dataset labels them as initially passing in the unmodified repo.
        self.p2p_status: Dict[str, bool] = {t: True for t in (self.pass_to_pass or [])}
        # Counters for diagnostics across the episode.
        self.total_solves = 0
        self.total_expansions = 0
        self.total_regressions = 0
        self.total_p2p_breaks = 0
        self.total_p2p_recovers = 0
        self.total_ticks = 0
        # Online-refinement state (prompt-only; see SchedulerConfig.refine_*).
        # `refined[t]` holds the decomposition (list of (name, code)) to render for t.
        # `refine_fn` is an optional callback(test_id) -> [(name, code)] injected by the
        # env; it MUST be pure (no scheduler-state mutation). None in unit tests / when
        # refinement is off. Nothing here is ever read by the reward computation.
        self.refined: Dict[str, list] = {}
        self.refine_fn = None
        self.total_refinements = 0

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _promote_from_pool(self, count: int) -> int:
        """Move up to `count` tests from POOL into ACTIVE (subject to m_max).

        Returns the number actually promoted.
        """
        room = max(0, self.config.m_max - len(self.active))
        n = min(count, room, len(self.pool))
        for _ in range(n):
            t = self.pool.pop(0)
            self.active[t] = 0
        return n

    def _tests_to_query(self) -> List[str]:
        """Tests whose status the caller should fetch this tick."""
        return list(self.active.keys()) + list(self.cooldown.keys())

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def tests_to_query(self) -> List[str]:
        """Public alias of `_tests_to_query` for the runtime."""
        return self._tests_to_query()

    def p2p_tests_to_query(self) -> List[str]:
        """All PASS_TO_PASS tests — runtime should re-check every k_steps tick.

        Per user-confirmed setting (2026-05-20), mid-episode runs all P2P
        each tick rather than sampling. This is expensive but faithful: a
        regression in *any* P2P breaks the patch under SWE-bench semantics.
        """
        return list(self.p2p_status.keys())

    def _update_p2p(
        self, p2p_pass_map: Optional[Dict[str, bool]]
    ) -> Tuple[int, int, int]:
        """Apply a P2P pass-map snapshot and return (newly_broken,
        newly_recovered, n_currently_broken).
        """
        if not p2p_pass_map:
            n_broken = sum(1 for ok in self.p2p_status.values() if not ok)
            return 0, 0, n_broken
        newly_broken = 0
        newly_recovered = 0
        for t, was_passing in list(self.p2p_status.items()):
            if t not in p2p_pass_map:
                continue
            is_passing = bool(p2p_pass_map[t])
            if was_passing and not is_passing:
                newly_broken += 1
                self.p2p_status[t] = False
            elif (not was_passing) and is_passing:
                newly_recovered += 1
                self.p2p_status[t] = True
        n_broken = sum(1 for ok in self.p2p_status.values() if not ok)
        return newly_broken, newly_recovered, n_broken

    def n_total(self) -> int:
        return len(self.fail_to_pass)

    def state_snapshot(self) -> Dict[str, List[str]]:
        return {
            "active": list(self.active.keys()),
            "cooldown": list(self.cooldown.keys()),
            "pool": list(self.pool),
            "retired": list(self.retired),
        }

    def tick(
        self,
        pass_map: Dict[str, bool],
        p2p_pass_map: Optional[Dict[str, bool]] = None,
    ) -> SchedulerEvent:
        """Advance the state machine using a fresh subset pass/fail map.

        `pass_map` only needs to contain entries for `tests_to_query()`; any
        missing key is treated as `failing` (conservative — the agent did not
        prove the test passes this tick).

        `p2p_pass_map`, if provided, is used to update P2P regression state
        and contribute a dense reward component. Pass `None` to skip P2P
        checks this tick (cheap mode).
        """
        self.total_ticks += 1
        cfg = self.config

        solves = 0
        expansions = 0
        regressions = 0
        promoted = 0
        refinements = 0   # telemetry only; never enters the dense reward below

        # --- Step 1: tick COOLDOWN and re-verify expired entries -------
        # Process cooldown FIRST so that tests promoted to cooldown later in
        # this same tick are not immediately decremented (semantics: a fresh
        # cooldown timer of D survives D subsequent ticks).
        expired: List[str] = []
        for t in list(self.cooldown.keys()):
            self.cooldown[t] -= 1
            if self.cooldown[t] <= 0:
                expired.append(t)
        for t in expired:
            del self.cooldown[t]
            still_green = bool(pass_map.get(t, False))
            if still_green:
                self.retired.append(t)
            else:
                # Regressed under cooldown -> back to ACTIVE w/ fresh counter.
                regressions += 1
                if len(self.active) < cfg.m_max:
                    self.active[t] = 0
                else:
                    # Active is full; push back into POOL (tail) so we revisit.
                    self.pool.append(t)

        # --- Step 2: process ACTIVE tests ------------------------------
        newly_cooldown: List[str] = []
        for t in list(self.active.keys()):
            passed = bool(pass_map.get(t, False))
            if passed:
                # Test solved this tick -> move to COOLDOWN.
                newly_cooldown.append(t)
                solves += 1
            else:
                # Increment stall counter; if we hit k_steps, expand POOL.
                self.active[t] += 1
                # --- Prompt-only refinement annotation (B1/R12) -------------
                # If this test has stalled for `refine_after` ticks, fetch its
                # cached decomposition ONCE and stash it for prompt_block() to
                # render. This sets `self.refined[t]` only; it does NOT touch the
                # stall-expand logic or any reward branch that follows, so the
                # reward stream is identical to refinement-off. `refine_fn` must
                # be pure (no scheduler mutation).
                if (
                    cfg.refine_enabled
                    and self.refine_fn is not None
                    and self.active[t] >= cfg.refine_after
                    and t not in self.refined
                ):
                    try:
                        subs = self.refine_fn(t)
                    except Exception:
                        subs = None
                    if subs:
                        self.refined[t] = list(subs)[: cfg.refine_max_subtests]
                        refinements += 1
                # -----------------------------------------------------------
                if self.active[t] >= cfg.k_steps:
                    self.active[t] = 0  # reset; expansion happens once per k window
                    n_promoted = self._promote_from_pool(cfg.expand_count)
                    if n_promoted > 0:
                        expansions += 1
                        promoted += n_promoted

        for t in newly_cooldown:
            self.active.pop(t, None)
            self.cooldown[t] = cfg.cooldown_steps

        # --- Step 3: don't starve. If ACTIVE empty + POOL non-empty, refill.
        if not self.active and self.pool:
            promoted += self._promote_from_pool(cfg.m_init)

        # --- Step 4: P2P state update (if a P2P map is supplied) -------
        p2p_newly_broken, p2p_newly_recovered, n_p2p_broken = self._update_p2p(p2p_pass_map)

        # --- Step 5: dense reward --------------------------------------
        n_active = len(self.active)
        r_alpha = cfg.alpha * solves
        r_beta = -cfg.beta * expansions
        r_delta = -cfg.delta * regressions
        r_gamma = -cfg.gamma
        r_eta = cfg.eta * (1.0 / (n_active + 1))
        r_p2p_break = -cfg.delta_p2p_break * p2p_newly_broken
        r_p2p_recover = cfg.delta_p2p_recover * p2p_newly_recovered
        reward = (
            r_alpha + r_beta + r_delta + r_gamma + r_eta
            + r_p2p_break + r_p2p_recover
        )

        self.total_solves += solves
        self.total_expansions += expansions
        self.total_regressions += regressions
        self.total_refinements += refinements
        self.total_p2p_breaks += p2p_newly_broken
        self.total_p2p_recovers += p2p_newly_recovered

        return SchedulerEvent(
            reward=reward,
            n_active=n_active,
            n_cooldown=len(self.cooldown),
            n_retired=len(self.retired),
            n_pool=len(self.pool),
            solves=solves,
            expansions=expansions,
            regressions=regressions,
            promoted=promoted,
            refinements=refinements,
            p2p_newly_broken=p2p_newly_broken,
            p2p_newly_recovered=p2p_newly_recovered,
            n_p2p_broken=n_p2p_broken,
            n_p2p_total=len(self.p2p_status),
            r_alpha=r_alpha,
            r_beta=r_beta,
            r_delta=r_delta,
            r_gamma=r_gamma,
            r_eta=r_eta,
            r_p2p_break=r_p2p_break,
            r_p2p_recover=r_p2p_recover,
        )

    def prompt_block(self, test_bodies: Optional[Dict[str, str]] = None) -> str:
        """Render the active-tests block for prompt injection.

        Keeps the block under `max_block_chars`. If `test_bodies` is given, a
        short snippet of each active test's source is included; otherwise just
        the test identifiers.
        """
        cfg = self.config
        if not self.active:
            return "<active_tests count=\"0\"/>"

        lines = [f"<active_tests count=\"{len(self.active)}\">"]
        budget = cfg.max_block_chars - len("\n".join(lines)) - len("</active_tests>")
        per_test = max(80, budget // max(1, len(self.active)))
        for t in self.active.keys():
            entry = f"  <test id=\"{t}\""
            if cfg.refine_enabled and self.refined.get(t):
                # Render the backtranslated decomposition as guidance for this
                # (historically hard) test. Prompt-only — never affects scoring.
                subs = self.refined[t]
                entry += " refined=\"true\">\n"
                entry += (
                    "    <!-- This test was hard to satisfy; here are smaller "
                    "sub-tests that together capture it. -->\n"
                )
                sub_budget = max(80, per_test // max(1, len(subs)))
                for name, code in subs:
                    snippet = (code or "").strip()
                    if len(snippet) > sub_budget:
                        snippet = snippet[:sub_budget] + "..."
                    entry += (
                        f"    <subtest name=\"{name}\">\n{snippet}\n"
                        f"    </subtest>\n"
                    )
                entry += "  </test>"
            elif test_bodies and t in test_bodies and cfg.show_bodies:
                body = test_bodies[t].strip()
                if len(body) > per_test - 40:
                    body = body[: per_test - 40] + "..."
                entry += f">\n    {body}\n  </test>"
            else:
                entry += "/>"
            lines.append(entry)
        lines.append("</active_tests>")
        out = "\n".join(lines)
        if len(out) > cfg.max_block_chars:
            out = out[: cfg.max_block_chars - 16] + "...</active_tests>"
        return out

    def terminal_reward(
        self,
        full_pass_map: Optional[Dict[str, bool]] = None,
        tokens_used: int = 0,
        p2p_pass_map: Optional[Dict[str, bool]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute the terminal reward from full FAIL_TO_PASS + PASS_TO_PASS maps.

        - `full_pass_map`: pass/fail per `fail_to_pass` test. None → fall back to
          (retired ∪ passing-cooldown).
        - `p2p_pass_map`: pass/fail per `pass_to_pass` test. None → fall back to
          internal `self.p2p_status` (the last seen state, updated by `tick()`).
          With v1 default (no mid-episode P2P ticks), this fallback is just the
          initial "all trusted" state, so frac_p2p_broken=0 — i.e. the env wrapper
          must supply a real terminal P2P map for the regression penalty to fire.

        Returns (total_reward, breakdown_dict). Breakdown keys:
        `frac_solved`, `r_solve`, `r_token`, `frac_p2p_broken`, `r_regress`,
        `tokens_used`.
        """
        cfg = self.config
        n = max(1, self.n_total())

        # FAIL_TO_PASS / solve credit
        if full_pass_map is None:
            solved = set(self.retired) | {t for t in self.cooldown.keys()}
        else:
            solved = {t for t, ok in full_pass_map.items() if ok}
        frac_solved = len(solved) / n
        r_solve = cfg.w_solve * frac_solved

        # PASS_TO_PASS / regression penalty
        n_p2p = len(self.p2p_status)
        if p2p_pass_map is not None:
            # Refresh internal state from the supplied map (only for keys present)
            for t in list(self.p2p_status.keys()):
                if t in p2p_pass_map:
                    self.p2p_status[t] = bool(p2p_pass_map[t])
            broken_set = {t for t, ok in self.p2p_status.items() if not ok}
        else:
            broken_set = {t for t, ok in self.p2p_status.items() if not ok}
        frac_p2p_broken = (len(broken_set) / n_p2p) if n_p2p > 0 else 0.0
        r_regress = -cfg.w_regress * frac_p2p_broken

        # Token penalty
        token_ratio = max(0, tokens_used) / max(1.0, cfg.token_ref)
        token_ratio = min(token_ratio, cfg.token_penalty_cap)  # clamp pathological outputs
        r_token = -cfg.lam_token * token_ratio

        r_final = r_solve + r_regress + r_token
        breakdown = {
            "frac_solved": frac_solved,
            "r_solve": r_solve,
            "frac_p2p_broken": frac_p2p_broken,
            "r_regress": r_regress,
            "r_token": r_token,
            "tokens_used": float(tokens_used),
            "n_p2p": float(n_p2p),
            "n_p2p_broken": float(len(broken_set)),
        }
        return r_final, breakdown
