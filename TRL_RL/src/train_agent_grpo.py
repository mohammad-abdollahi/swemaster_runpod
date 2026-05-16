"""Minimal multi-turn GRPO trainer for the SWE agent task.

Single-GPU, no Ray, no TRL (TRL's GRPOTrainer is single-shot only — the
multi-turn rollout breaks its assumptions).  REINFORCE / GRPO with
leave-one-out advantage:

    A_i = R_i - (1/(G-1)) sum_{j != i} R_j

Loss is computed only on assistant tokens; tool-result (user) tokens and
system/prompt tokens are masked out.  Logs per-step metrics to stdout
and TensorBoard.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer

# Allow sibling imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent_rollout import rollout, RolloutResult  # noqa: E402
from data import swe_bench_demo  # noqa: E402
from swe_test_reward import compute_test_reward  # noqa: E402
from rewards import _extract_diff, _flatten  # noqa: E402
from swe_sandbox import SweSandbox  # noqa: E402

import difflib  # noqa: E402


# --------------------------------------------------------------------------
# args
# --------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--logging-dir", required=True)
    p.add_argument("--swe-parquet", default=None)
    p.add_argument("--max-steps", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=2,
                   help="Distinct prompts per training step.")
    p.add_argument("--num-generations", type=int, default=2,
                   help="GRPO group size — rollouts per prompt.")
    p.add_argument("--max-prompt-tokens", type=int, default=4096)
    p.add_argument("--max-gen-tokens", type=int, default=512,
                   help="Cap on assistant tokens per turn.")
    p.add_argument("--max-turns", type=int, default=6)
    p.add_argument("--learning-rate", type=float, default=1e-6)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--reward", choices=["proxy", "test"], default="proxy",
                   help="'proxy' = diff/diff SequenceMatcher (no env). "
                        "'test' = paper-faithful binary SWE-bench reward "
                        "(needs 04_setup_test_envs.sh).")
    p.add_argument("--test-timeout", type=int, default=600)
    return p.parse_args()


# --------------------------------------------------------------------------
# rollout helpers
# --------------------------------------------------------------------------
def make_generate_fn(model, tokenizer, max_new_tokens: int, temperature: float):
    """Return a ``messages -> str`` callable bound to ``model``."""
    device = next(model.parameters()).device
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    def _generate(messages):
        # Render the chat template as text first, then tokenize — avoids the
        # BatchEncoding-vs-tensor API split between transformers versions.
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = enc.input_ids.to(device)
        with torch.inference_mode():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=enc.attention_mask.to(device),
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-6),
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )
        gen = out[0, input_ids.shape[1]:]
        return tokenizer.decode(gen, skip_special_tokens=True)

    return _generate


def patch_reward(candidate: str, gold: str) -> float:
    """Same proxy as ``swe_patch_similarity`` — diff/diff SequenceMatcher."""
    pred = _extract_diff(candidate)
    return float(difflib.SequenceMatcher(None, pred, gold or "").ratio())


# --------------------------------------------------------------------------
# loss computation
# --------------------------------------------------------------------------
def tokenize_trajectory(
    tokenizer,
    messages: list[dict[str, str]],
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize the full transcript and return (input_ids, assistant_mask).

    ``assistant_mask[i] == 1`` for tokens that the model itself produced (so
    the gradient flows there) and ``0`` for prompt / tool-result tokens.

    The chat template typically wraps each turn in role markers; we tokenize
    each message individually and concatenate so we can track ownership.
    """
    ids_chunks: list[torch.Tensor] = []
    mask_chunks: list[torch.Tensor] = []

    # We need a trailing add_generation_prompt=False render of the whole
    # transcript so EOS / role tokens line up the same way they would at
    # generation time.  Easiest is per-message rendering with template.
    for i, msg in enumerate(messages):
        is_last = (i == len(messages) - 1)
        rendered = tokenizer.apply_chat_template(
            [msg],
            tokenize=False,
            add_generation_prompt=False,
        )
        chunk_ids = tokenizer(rendered, add_special_tokens=False,
                              return_tensors="pt").input_ids[0]
        if i == 0 and tokenizer.bos_token_id is not None and chunk_ids[0] != tokenizer.bos_token_id:
            pass  # template handles BOS internally for Qwen3
        ids_chunks.append(chunk_ids)
        mask_chunks.append(torch.full_like(
            chunk_ids,
            1 if msg["role"] == "assistant" else 0,
        ))

    ids = torch.cat(ids_chunks)
    mask = torch.cat(mask_chunks)
    if len(ids) > max_length:
        ids = ids[-max_length:]
        mask = mask[-max_length:]
    return ids, mask


def backward_per_trajectory(
    model,
    tokenizer,
    trajectories: list[tuple[list[dict[str, str]], float]],  # (messages, advantage)
    max_length: int,
    device: str,
) -> dict[str, float]:
    """Process each trajectory separately, scale loss by 1/N, call backward,
    then drop the forward graph.  Equivalent to mean-loss + single backward,
    but the activation memory of only ONE trajectory's forward pass is alive
    at a time — critical for multi-turn rollouts where a single trajectory
    can be 10k+ tokens.
    """
    n_valid = 0
    # First pass: count valid trajectories so we can scale 1/N up front.
    cached: list[tuple[torch.Tensor, torch.Tensor, float]] = []
    for messages, advantage in trajectories:
        ids, mask = tokenize_trajectory(tokenizer, messages, max_length)
        if mask.sum().item() == 0:
            continue
        cached.append((ids, mask, advantage))
        n_valid += 1
    if n_valid == 0:
        return {"loss": 0.0, "assistant_tokens": 0}

    scale = 1.0 / n_valid
    total_loss = 0.0
    total_tokens = 0
    for ids, mask, advantage in cached:
        ids = ids.to(device).unsqueeze(0)
        mask = mask.to(device).unsqueeze(0)
        outputs = model(input_ids=ids, use_cache=False)
        logits = outputs.logits[:, :-1, :]
        targets = ids[:, 1:]
        target_mask = mask[:, 1:].float()
        logp = F.log_softmax(logits, dim=-1)
        token_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        num = target_mask.sum().clamp(min=1.0)
        mean_logp = (token_logp * target_mask).sum() / num
        traj_loss = -advantage * mean_logp * scale
        traj_loss.backward()
        total_loss += float(traj_loss.item())
        total_tokens += int(num.item())
        # Drop large intermediates before next traj's forward.
        del outputs, logits, logp, token_logp, mean_logp, traj_loss
        torch.cuda.empty_cache()
    return {"loss": total_loss, "assistant_tokens": total_tokens}


# --------------------------------------------------------------------------
# training loop
# --------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    print(f"[agent-grpo] model={args.model_path}")
    print(f"[agent-grpo] output={args.output_dir}")
    print(f"[agent-grpo] tensorboard={args.logging_dir}")
    print(f"[agent-grpo] max_steps={args.max_steps} bs={args.batch_size} "
          f"num_gen={args.num_generations} max_turns={args.max_turns}")

    device = "cuda:0"
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        trust_remote_code=True,
    ).to(device)
    model.gradient_checkpointing_enable()
    model.train()  # we need grads for the loss, but rollouts will toggle eval

    ds = swe_bench_demo(args.swe_parquet)
    print(f"[agent-grpo] dataset rows: {len(ds)}; cols: {ds.column_names}")
    n = len(ds)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
    )
    writer = SummaryWriter(args.logging_dir)

    generate_fn = make_generate_fn(
        model, tokenizer, args.max_gen_tokens, args.temperature,
    )

    global_step = 0
    for step in range(args.max_steps):
        step_start = time.time()
        # Pick prompts for this step (round-robin over the tiny demo set).
        prompt_indices = [(step * args.batch_size + i) % n
                          for i in range(args.batch_size)]

        trajectories_for_loss: list[tuple[list[dict[str, str]], float]] = []
        rewards_log: list[float] = []
        num_submissions = 0
        turn_counts: list[int] = []
        candidate_lens: list[int] = []

        # Switch to eval mode for rollouts (faster, no grad).
        model.eval()
        rollouts: list[tuple[int, RolloutResult, float]] = []  # (prompt_idx, r, reward)
        for p_idx in prompt_indices:
            row = ds[p_idx]
            sandbox = SweSandbox(
                repo=row["repo"],
                base_commit=row["base_commit"],
                gold_patch=row["gold_patch"],
                instance_id=row["instance_id"],
            )
            for g in range(args.num_generations):
                r = rollout(
                    sandbox=sandbox,
                    user_problem=row["problem_statement"],
                    generate_fn=generate_fn,
                    max_turns=args.max_turns,
                )
                # IMPORTANT: compute reward NOW, before the next generation's
                # sandbox.reset() wipes this rollout's worktree edits.
                if args.reward == "test":
                    reward, detail = compute_test_reward(
                        sandbox, row, timeout=args.test_timeout,
                    )
                    if g == 0:
                        print(f"  [test-reward {row['instance_id']}] "
                              f"R={reward} {detail[:160]}")
                else:
                    reward = patch_reward(r.candidate_patch, row["gold_patch"])
                rewards_log.append(reward)
                turn_counts.append(r.num_turns)
                candidate_lens.append(len(r.candidate_patch))
                if r.submitted:
                    num_submissions += 1
                rollouts.append((p_idx, r, reward))

        # Group-relative advantages (leave-one-out within each prompt's group).
        for p_idx in prompt_indices:
            group = [(r, rw) for pi, r, rw in rollouts if pi == p_idx]
            if len(group) < 2:
                # No meaningful advantage with a single rollout; use raw R - 0.
                for r, rw in group:
                    trajectories_for_loss.append((r.messages, rw))
                continue
            rewards = [rw for _, rw in group]
            for i, (r, rw) in enumerate(group):
                others = [rewards[j] for j in range(len(group)) if j != i]
                advantage = rw - sum(others) / len(others)
                trajectories_for_loss.append((r.messages, advantage))

        # Switch back to train mode for the gradient update.
        model.train()
        optimizer.zero_grad()
        # Memory-thrifty: process one trajectory at a time, accumulating grads.
        loss_info = backward_per_trajectory(
            model, tokenizer, trajectories_for_loss,
            max_length=args.max_prompt_tokens, device=device,
        )
        if loss_info["assistant_tokens"] > 0:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            optimizer.step()
        else:
            grad_norm = 0.0

        # Metrics
        mean_reward = sum(rewards_log) / max(len(rewards_log), 1)
        max_reward = max(rewards_log) if rewards_log else 0.0
        reward_std = (
            (sum((r - mean_reward) ** 2 for r in rewards_log) / len(rewards_log)) ** 0.5
            if rewards_log else 0.0
        )
        elapsed = time.time() - step_start

        metrics = {
            "train/reward_mean": mean_reward,
            "train/reward_max": max_reward,
            "train/reward_std": reward_std,
            "train/loss": loss_info["loss"],
            "train/grad_norm": grad_norm,
            "train/assistant_tokens": float(loss_info["assistant_tokens"]),
            "train/num_submissions": float(num_submissions),
            "train/mean_num_turns": (sum(turn_counts) / max(len(turn_counts), 1)),
            "train/mean_candidate_chars": (sum(candidate_lens) / max(len(candidate_lens), 1)),
            "train/step_time_s": elapsed,
        }
        msg = " ".join(f"{k.split('/')[-1]}={v:.4f}" for k, v in metrics.items())
        print(f"[step {step+1}/{args.max_steps}] {msg}")
        for k, v in metrics.items():
            writer.add_scalar(k, v, global_step)
        global_step += 1
        writer.flush()

    writer.close()
    # Save model state for inspection
    ckpt = Path(args.output_dir) / "final"
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tokenizer.save_pretrained(ckpt)
    print(f"[agent-grpo] DONE — checkpoint at {ckpt}")


if __name__ == "__main__":
    main()
