"""TRL GRPO smoke training.

Designed to run on the Trillium login node where ``ulimit -u = 1024`` makes
Ray-based trainers (verl / rllm) impossible. Uses HF Accelerate, no Ray,
no vLLM async — just enough machinery for GRPO to do rollout → reward →
advantage → loss → log.

Per-step metrics are emitted to stdout AND TensorBoard event files under
``--logging-dir``.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# Make sibling modules importable when this file is run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import yes_synthetic, gsm8k_subset, swe_bench_demo  # noqa: E402
from rewards import yes_token_reward, gsm8k_numeric_match, swe_patch_similarity  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True,
                   help="Local HF snapshot dir, e.g. Qwen3-4B snapshot.")
    p.add_argument("--output-dir", required=True,
                   help="Where TRL writes checkpoints & state.")
    p.add_argument("--logging-dir", required=True,
                   help="TensorBoard event files go here.")
    p.add_argument("--dataset", choices=["yes", "gsm8k", "swe"], default="yes",
                   help="'swe' loads the SWE-Master demo RL parquet and uses a "
                        "diff-similarity proxy reward (no docker required).")
    p.add_argument("--swe-parquet", default=None,
                   help="Path to the SWE-bench parquet (when --dataset=swe).")
    p.add_argument("--max-steps", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=2,
                   help="Per-device train batch (prompts per step).")
    p.add_argument("--num-generations", type=int, default=4,
                   help="GRPO group size — completions sampled per prompt.")
    # TRL 0.29 removed max_prompt_length — prompt truncation is handled by the
    # tokenizer's model_max_length. We pass --max-prompt-length so we can clamp
    # the tokenizer; the GRPOConfig only consumes max_completion_length.
    p.add_argument("--max-prompt-length", type=int, default=128)
    p.add_argument("--max-completion-length", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-6)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--use-vllm", action="store_true",
                   help="Use vLLM for rollout. Off by default to keep "
                        "process count low on the login node.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"[trl-grpo] model={args.model_path}")
    print(f"[trl-grpo] dataset={args.dataset}  max_steps={args.max_steps}")
    print(f"[trl-grpo] output={args.output_dir}")
    print(f"[trl-grpo] tensorboard={args.logging_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = args.max_prompt_length

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        trust_remote_code=True,
    )

    if args.dataset == "yes":
        train_dataset = yes_synthetic(n=max(64, args.batch_size * args.max_steps * 4))
        reward_funcs = yes_token_reward
    elif args.dataset == "gsm8k":
        train_dataset = gsm8k_subset(n=max(64, args.batch_size * args.max_steps * 4))
        reward_funcs = gsm8k_numeric_match
    else:  # swe
        train_dataset = swe_bench_demo(args.swe_parquet)
        reward_funcs = swe_patch_similarity
        print(f"[trl-grpo] loaded SWE dataset: {len(train_dataset)} rows; "
              f"columns: {train_dataset.column_names}")

    config = GRPOConfig(
        output_dir=args.output_dir,
        logging_dir=args.logging_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        logging_steps=1,
        save_steps=max(args.max_steps, 1) * 100,  # don't checkpoint during smoke
        save_total_limit=1,
        bf16=args.bf16,
        report_to=["tensorboard"],
        use_vllm=args.use_vllm,
        # Keep dataloader workers low to avoid thread/process pressure.
        dataloader_num_workers=0,
        # GRPO defaults a high beta; smoke is fine with the default.
        seed=0,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    print("[trl-grpo] starting trainer.train() ...")
    trainer.train()
    print("[trl-grpo] DONE")


if __name__ == "__main__":
    main()
