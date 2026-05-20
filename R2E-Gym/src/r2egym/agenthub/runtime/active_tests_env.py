"""Integration glue between `ActiveTestScheduler` and the SWE env wrapper.

The scheduler module (`active_tests.py`) is pure-Python and unit-tested. This
file owns the *side-effects* — reading env vars to build a `SchedulerConfig`,
holding the scheduler against an env instance, calling the Docker runtime to
fetch pass maps, and rendering the prompt block onto the task instruction.

Lifecycle (called from `rllm.environments.swe.swe_arsenal.SWEEnv`):

    state = ActiveTestState.maybe_create(runtime)
    instruction = state.augment_instruction(instruction)
    # ... agent rollout ...
    state.note_observation_tokens(obs)        # called every step
    state.maybe_tick(runtime)                 # called every step; gated
    reward, info = state.terminal_reward(runtime)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from r2egym.agenthub.runtime.active_tests import (
    ActiveTestScheduler,
    SchedulerConfig,
    SchedulerEvent,
)


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    try:
        return float(val)
    except ValueError:
        return default


def env_enabled() -> bool:
    """Master switch — when False the wrapper is a complete no-op."""
    return _env_bool("RLLM_ACTIVE_TESTS_ENABLED", True)


def mid_episode_enabled() -> bool:
    """Whether to run subset pytest every k_steps (v2). Default off (v1)."""
    return _env_bool("RLLM_ACTIVE_TESTS_MID_EPISODE", False)


def config_from_env() -> SchedulerConfig:
    """Build a `SchedulerConfig` from `RLLM_ACTIVE_TESTS_*` env vars."""
    cfg = SchedulerConfig()
    cfg.k_steps = _env_int("RLLM_ACTIVE_TESTS_K", cfg.k_steps)
    cfg.cooldown_steps = _env_int("RLLM_ACTIVE_TESTS_COOLDOWN", cfg.cooldown_steps)
    cfg.m_init = _env_int("RLLM_ACTIVE_TESTS_M_INIT", cfg.m_init)
    cfg.m_max = _env_int("RLLM_ACTIVE_TESTS_M_MAX", cfg.m_max)
    cfg.expand_count = _env_int("RLLM_ACTIVE_TESTS_EXPAND", cfg.expand_count)
    cfg.alpha = _env_float("RLLM_ACTIVE_TESTS_ALPHA", cfg.alpha)
    cfg.beta = _env_float("RLLM_ACTIVE_TESTS_BETA", cfg.beta)
    cfg.delta = _env_float("RLLM_ACTIVE_TESTS_DELTA", cfg.delta)
    cfg.gamma = _env_float("RLLM_ACTIVE_TESTS_GAMMA", cfg.gamma)
    cfg.eta = _env_float("RLLM_ACTIVE_TESTS_ETA", cfg.eta)
    cfg.w_solve = _env_float("RLLM_ACTIVE_TESTS_W_SOLVE", cfg.w_solve)
    cfg.lam_token = _env_float("RLLM_ACTIVE_TESTS_LAM_TOKEN", cfg.lam_token)
    cfg.token_ref = _env_float("RLLM_ACTIVE_TESTS_TOKEN_REF", cfg.token_ref)
    cfg.max_block_chars = _env_int("RLLM_ACTIVE_TESTS_BLOCK_CHARS", cfg.max_block_chars)
    return cfg


def _approx_tokens(text: str) -> int:
    """Rough token count for the token-penalty term.

    Uses 4 chars/token, which is within a factor of ~1.3 of any real BPE
    tokenizer on English+code. Exactness is unimportant — we only need a
    monotone signal proportional to how much context the agent burned.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


@dataclass
class ActiveTestState:
    """Per-rollout scheduler state attached to an `SWEEnv` instance.

    The class is intentionally fault-tolerant: if anything raises (the
    runtime is missing a method, the dataset has no FAIL_TO_PASS, pytest
    times out, etc.), we degrade gracefully to "no scheduler" and the
    legacy reward path takes over.
    """

    scheduler: Optional[ActiveTestScheduler] = None
    config: SchedulerConfig = field(default_factory=SchedulerConfig)
    tokens_used: int = 0
    step_count: int = 0
    tick_count: int = 0
    last_tick_step: int = 0
    enabled: bool = True
    mid_episode: bool = False
    last_event: Optional[SchedulerEvent] = None
    log_prefix: str = "[active-tests]"

    @classmethod
    def maybe_create(
        cls,
        runtime,
        log_prefix: str = "[active-tests]",
    ) -> Optional["ActiveTestState"]:
        """Build a state object if the master switch is on AND the runtime
        exposes a FAIL_TO_PASS list. Otherwise return None.
        """
        if not env_enabled():
            return None
        try:
            fail_to_pass = runtime.get_fail_to_pass_ids()
        except Exception:
            fail_to_pass = []
        if not fail_to_pass:
            return None
        cfg = config_from_env()
        try:
            sched = ActiveTestScheduler(fail_to_pass=fail_to_pass, config=cfg)
        except Exception as e:
            print(f"{log_prefix} scheduler init failed: {e!r}; disabling")
            return None
        return cls(
            scheduler=sched,
            config=cfg,
            enabled=True,
            mid_episode=mid_episode_enabled(),
            log_prefix=log_prefix,
        )

    # ------------------------------------------------------------------
    # prompt
    # ------------------------------------------------------------------
    def augment_instruction(self, instruction: str) -> str:
        """Append the `<active_tests>` block to the agent's task instruction.

        Includes a short natural-language preamble so the agent knows the
        block is hint information about what success looks like.
        """
        if not self._is_live():
            return instruction
        block = self.scheduler.prompt_block()
        if not block or block == "<active_tests count=\"0\"/>":
            return instruction
        preamble = (
            "\n\n--- Active test targets ---\n"
            "The following pytest node ids are an initial subset of the hidden "
            "test set you need to make pass. Running them inside the container "
            "(via pytest) is a fast way to check your progress; the final "
            "reward is computed against the FULL test set, so do not overfit "
            "to just these.\n"
        )
        # Tokens consumed by the injected block also count towards the budget.
        self.tokens_used += _approx_tokens(preamble + block)
        return instruction + preamble + block

    # ------------------------------------------------------------------
    # per-step
    # ------------------------------------------------------------------
    def note_observation_tokens(self, obs: object) -> None:
        try:
            self.tokens_used += _approx_tokens(str(obs))
        except Exception:
            pass
        self.step_count += 1

    def maybe_tick(self, runtime) -> Optional[SchedulerEvent]:
        """Run a scheduler tick if mid-episode mode is on AND we're at a
        k_steps boundary. Returns the event or None.
        """
        if not (self._is_live() and self.mid_episode):
            return None
        if self.step_count - self.last_tick_step < self.config.k_steps:
            return None
        t0 = time.time()
        try:
            pass_map = runtime.get_fail_to_pass_status(timeout=300)
        except Exception as e:
            print(f"{self.log_prefix} mid-episode pass_map failed: {e!r}")
            pass_map = {}
        evt = self.scheduler.tick(pass_map or {})
        self.last_tick_step = self.step_count
        self.tick_count += 1
        self.last_event = evt
        print(
            f"{self.log_prefix} tick {self.tick_count} @step{self.step_count} "
            f"reward={evt.reward:+.3f} solves={evt.solves} expand={evt.expansions} "
            f"regress={evt.regressions} n_active={evt.n_active} "
            f"({time.time() - t0:.1f}s)"
        )
        return evt

    # ------------------------------------------------------------------
    # terminal
    # ------------------------------------------------------------------
    def terminal_reward(self, runtime) -> Tuple[float, Dict[str, float]]:
        """Compute the trajectory's terminal reward.

        Returns `(reward, info)`. If the scheduler is not live we still
        gracefully compute the legacy binary reward by calling the runtime's
        ordinary `_calculate_reward` — but in that case we are usually not
        called at all (the caller decides).
        """
        info: Dict[str, float] = {"tokens_used": float(self.tokens_used)}
        if not self._is_live():
            return 0.0, info
        try:
            pass_map = runtime.get_fail_to_pass_status(timeout=600)
        except Exception as e:
            print(f"{self.log_prefix} terminal pass_map failed: {e!r}")
            pass_map = {}
        reward, breakdown = self.scheduler.terminal_reward(
            full_pass_map=pass_map,
            tokens_used=self.tokens_used,
        )
        info.update(breakdown)
        info["n_total"] = float(self.scheduler.n_total())
        info["n_retired"] = float(len(self.scheduler.retired))
        info["n_active"] = float(len(self.scheduler.active))
        info["n_cooldown"] = float(len(self.scheduler.cooldown))
        info["total_solves"] = float(self.scheduler.total_solves)
        info["total_regressions"] = float(self.scheduler.total_regressions)
        info["total_expansions"] = float(self.scheduler.total_expansions)
        info["tick_count"] = float(self.tick_count)
        print(
            f"{self.log_prefix} terminal reward={reward:+.3f} "
            f"frac_solved={breakdown.get('frac_solved', 0.0):.3f} "
            f"tokens={self.tokens_used} ticks={self.tick_count}"
        )
        return float(reward), info

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _is_live(self) -> bool:
        return self.enabled and self.scheduler is not None
