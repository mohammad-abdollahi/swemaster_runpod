"""Reward functions for the TRL_RL smoke pipeline.

Each function follows the TRL GRPOTrainer signature:
    reward_fn(prompts, completions, **kwargs) -> list[float]
"""
from __future__ import annotations

import difflib
import re
from typing import Any


def yes_token_reward(
    prompts: list[Any],
    completions: list[Any],
    **_: Any,
) -> list[float]:
    """1.0 if the completion contains the literal token YES, else 0.0.

    Intentionally trivial: it gives a non-degenerate reward signal in the
    first rollout so we see real GRPO advantages/loss values, not all-zero.
    """
    out: list[float] = []
    for c in completions:
        text = _flatten(c)
        out.append(1.0 if "YES" in text.upper() else 0.0)
    return out


_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def gsm8k_numeric_match(
    prompts: list[Any],
    completions: list[Any],
    answer: list[str] | None = None,
    **_: Any,
) -> list[float]:
    """Reward 1.0 if the last number in the completion matches the gold answer.

    Used when training on GSM8K-style problems. The dataset must include an
    ``answer`` column with the gold numeric string.
    """
    if answer is None:
        return [0.0] * len(completions)
    rewards: list[float] = []
    for comp, gold in zip(completions, answer):
        text = _flatten(comp)
        nums = _NUMBER_RE.findall(text)
        pred = nums[-1] if nums else None
        rewards.append(1.0 if pred is not None and _close(pred, gold) else 0.0)
    return rewards


_DIFF_BLOCK_RE = re.compile(
    r"(diff --git .*?)(?=\ndiff --git |\Z)",
    re.DOTALL,
)


def _extract_diff(text: str) -> str:
    """Pull the diff block out of a model completion.

    The model may surround the diff with prose, code fences, etc.  We grab
    everything from the first ``diff --git`` to either the next one or end.
    """
    fenced = re.search(r"```(?:diff)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    blocks = _DIFF_BLOCK_RE.findall(text)
    return "\n".join(blocks) if blocks else text


def swe_patch_similarity(
    prompts: list[Any],
    completions: list[Any],
    gold_patch: list[str] | None = None,
    **_: Any,
) -> list[float]:
    """Proxy reward: SequenceMatcher ratio between model's diff and gold.

    Range ``[0, 1]``. Stand-in for the real "apply patch, run tests" reward
    until the Apptainer SWE-bench backend is in place.  Two diffs that touch
    the same files & lines score high even if the wording differs slightly,
    so the RL gradient still points the model toward sensible diffs.
    """
    if gold_patch is None:
        return [0.0] * len(completions)
    out: list[float] = []
    for comp, gold in zip(completions, gold_patch):
        pred = _extract_diff(_flatten(comp))
        ratio = difflib.SequenceMatcher(None, pred, gold or "").ratio()
        out.append(float(ratio))
    return out


def _flatten(completion: Any) -> str:
    """TRL passes completions either as raw strings or chat-formatted lists."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        return "".join(turn.get("content", "") for turn in completion if isinstance(turn, dict))
    return str(completion)


def _close(pred: str, gold: str) -> bool:
    try:
        return abs(float(pred) - float(gold)) < 1e-4
    except ValueError:
        return pred.strip() == gold.strip()
