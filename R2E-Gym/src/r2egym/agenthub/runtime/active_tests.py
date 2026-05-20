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

    # Dense reward weights
    alpha: float = 1.0           # per newly-solved test (the strongest signal)
    beta: float = 0.2            # per stall-expansion (cost of getting stuck)
    delta: float = 0.5           # per regression on cooldown re-check
    gamma: float = 0.01          # per scheduler tick (per eval cadence)
    eta: float = 0.05            # weight on the 1/(|ACTIVE|+1) shaping term

    # Terminal reward weights (applied separately in the env wrapper)
    w_solve: float = 5.0         # multiplier on (|RETIRED ∪ COOLDOWN_passing|)/N
    lam_token: float = 0.1       # multiplier on tokens_used/token_ref
    token_ref: float = 32000.0   # token-penalty reference

    # Prompt formatting
    max_block_chars: int = 4000  # hard cap on the injected prompt-block size
    show_bodies: bool = False    # if True, scheduler caller can attach test bodies

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


@dataclass
class SchedulerEvent:
    """Per-tick result returned to the env."""

    reward: float
    n_active: int
    n_cooldown: int
    n_retired: int
    n_pool: int
    # diagnostics
    solves: int = 0          # tests that turned green this tick
    expansions: int = 0      # number of stall-expand events
    regressions: int = 0     # cooldown re-checks that regressed
    promoted: int = 0        # individual tests pulled from POOL to ACTIVE
    # shape components (for logging)
    r_alpha: float = 0.0
    r_beta: float = 0.0
    r_delta: float = 0.0
    r_gamma: float = 0.0
    r_eta: float = 0.0


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

    def __post_init__(self) -> None:
        self.active: "OrderedDict[str, int]" = OrderedDict()  # test -> stall counter
        self.pool: List[str] = list(self.fail_to_pass)
        self.cooldown: "OrderedDict[str, int]" = OrderedDict()  # test -> steps remaining
        self.retired: List[str] = []
        # Promote the initial ACTIVE window.
        self._promote_from_pool(self.config.m_init)
        # Counters for diagnostics across the episode.
        self.total_solves = 0
        self.total_expansions = 0
        self.total_regressions = 0
        self.total_ticks = 0

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

    def n_total(self) -> int:
        return len(self.fail_to_pass)

    def state_snapshot(self) -> Dict[str, List[str]]:
        return {
            "active": list(self.active.keys()),
            "cooldown": list(self.cooldown.keys()),
            "pool": list(self.pool),
            "retired": list(self.retired),
        }

    def tick(self, pass_map: Dict[str, bool]) -> SchedulerEvent:
        """Advance the state machine using a fresh subset pass/fail map.

        `pass_map` only needs to contain entries for `tests_to_query()`; any
        missing key is treated as `failing` (conservative — the agent did not
        prove the test passes this tick).
        """
        self.total_ticks += 1
        cfg = self.config

        solves = 0
        expansions = 0
        regressions = 0
        promoted = 0

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

        # --- Step 4: dense reward --------------------------------------
        n_active = len(self.active)
        r_alpha = cfg.alpha * solves
        r_beta = -cfg.beta * expansions
        r_delta = -cfg.delta * regressions
        r_gamma = -cfg.gamma
        r_eta = cfg.eta * (1.0 / (n_active + 1))
        reward = r_alpha + r_beta + r_delta + r_gamma + r_eta

        self.total_solves += solves
        self.total_expansions += expansions
        self.total_regressions += regressions

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
            r_alpha=r_alpha,
            r_beta=r_beta,
            r_delta=r_delta,
            r_gamma=r_gamma,
            r_eta=r_eta,
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
            if test_bodies and t in test_bodies and cfg.show_bodies:
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
    ) -> Tuple[float, Dict[str, float]]:
        """Compute the terminal reward from a *full* pass/fail map over `fail_to_pass`.

        If `full_pass_map` is None we fall back to (retired ∪ passing-cooldown)
        as the "solved" set — useful when the caller cannot afford a final
        full-suite run.
        """
        cfg = self.config
        n = max(1, self.n_total())

        if full_pass_map is None:
            solved = set(self.retired) | {
                t for t in self.cooldown.keys()  # presumed passing while in cooldown
            }
        else:
            solved = {t for t, ok in full_pass_map.items() if ok}

        frac = len(solved) / n
        r_solve = cfg.w_solve * frac
        r_token = -cfg.lam_token * (max(0, tokens_used) / max(1.0, cfg.token_ref))
        r_final = r_solve + r_token
        breakdown = {
            "frac_solved": frac,
            "r_solve": r_solve,
            "r_token": r_token,
            "tokens_used": float(tokens_used),
        }
        return r_final, breakdown
