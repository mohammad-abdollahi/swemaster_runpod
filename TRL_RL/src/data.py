"""Dataset builders for the TRL_RL smoke pipeline.

Datasets are returned as plain HuggingFace ``Dataset`` objects with a
``prompt`` column (and ``answer`` where applicable). TRL's GRPOTrainer
auto-tokenizes them.
"""
from __future__ import annotations

import random
from datasets import Dataset


def yes_synthetic(n: int = 512, seed: int = 0) -> Dataset:
    """Trivial verifiable task: ask the model to emit ``YES``.

    Variety in the prompt wording keeps the model from memorizing a single
    token sequence — we still want a real distribution over rollouts.
    """
    rng = random.Random(seed)
    templates = [
        "Reply with the single word YES.",
        "Are you ready? Answer YES.",
        "If you understand, say YES.",
        "Please type YES.",
        "To confirm, output YES and nothing else.",
        "Just respond with: YES",
    ]
    prompts = [rng.choice(templates) for _ in range(n)]
    return Dataset.from_dict({"prompt": prompts})


_SWE_SYSTEM_PROMPT = (
    "You are an expert software engineer. When given a repository name and an "
    "issue description, think briefly about the root cause, then produce ONE "
    "unified diff patch (in `diff --git` format) that resolves the issue. "
    "Always end your response with the patch."
)

_SWE_USER_TEMPLATE = """\
Repository: {repo}

Issue:
{problem_statement}

Produce the final patch now (`diff --git ...`)."""


def swe_bench_demo(parquet_path: str | None = None) -> Dataset:
    """Load the local SWE-bench-style demo RL dataset.

    The parquet ships with the repo at:
        data_examples/rl_data/ood_data/.../train_00000.parquet
    Each row is a real SWE-bench task. Columns we use:
      - ``problem_statement``  : GitHub issue text  → prompt
      - ``repo``                : repo name        → prompt
      - ``patch``               : gold unified diff → reward target
    Other columns (``test_patch``, ``docker_image``, ``FAIL_TO_PASS``, ...)
    are needed for the *real* test-based reward and ignored here.
    """
    import os
    import pandas as pd
    if parquet_path is None:
        parquet_path = os.environ.get(
            "SWE_RL_PARQUET",
            "data_examples/rl_data/ood_data/equal_0.3999999_0.9999999_"
            "swe-gym_and_rebench_demo_for_RL/train_00000.parquet",
        )
    df = pd.read_parquet(parquet_path)
    # Conversational format — TRL recognizes `prompt` as a list of messages
    # and applies the tokenizer's chat template automatically. Important for
    # the SWE-Master-4B-SFT model, which was SFT'd with the Qwen chat template.
    prompts = [
        [
            {"role": "system", "content": _SWE_SYSTEM_PROMPT},
            {"role": "user", "content": _SWE_USER_TEMPLATE.format(
                repo=row["repo"],
                problem_statement=row["problem_statement"],
            )},
        ]
        for _, row in df.iterrows()
    ]
    # Carry the SWE-bench test-reward fields too (as strings — HF Datasets
    # needs homogeneous columns and the parquet stores these as numpy arrays
    # / JSON; swe_test_reward._as_list parses them back).
    def _s(col):
        return [str(x) for x in df[col].tolist()]

    return Dataset.from_dict({
        "prompt": prompts,
        "gold_patch": df["patch"].tolist(),
        "instance_id": df["instance_id"].tolist(),
        # Needed by the agent rollout to clone+reset the right working tree.
        "repo": df["repo"].tolist(),
        "base_commit": df["base_commit"].tolist(),
        "problem_statement": df["problem_statement"].tolist(),
        # Needed by the paper-faithful test reward.
        "test_patch": _s("test_patch"),
        "make_test_spec": _s("make_test_spec"),
        "FAIL_TO_PASS": _s("FAIL_TO_PASS"),
        "PASS_TO_PASS": _s("PASS_TO_PASS"),
    })


def gsm8k_subset(n: int = 256, split: str = "train") -> Dataset:
    """GSM8K subset for a slightly less trivial smoke task.

    Only used when ``--dataset gsm8k`` is passed; not the default to keep the
    smoke fast and avoid an extra HF download.
    """
    from datasets import load_dataset
    raw = load_dataset("gsm8k", "main", split=split)
    raw = raw.shuffle(seed=0).select(range(min(n, len(raw))))

    def _to_prompt_answer(ex):
        # gold answer comes after the literal "#### " token in GSM8K
        gold = ex["answer"].split("####")[-1].strip()
        return {
            "prompt": ex["question"].strip() + "\nLet's reason step by step.",
            "answer": gold,
        }

    return raw.map(_to_prompt_answer, remove_columns=raw.column_names)
