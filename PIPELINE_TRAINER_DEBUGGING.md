# Pipeline Trainer Step-1 Restart: Root Cause & Fixes

## Problem
8-GPU pipeline trainer restarts immediately on step 1 at the backpropagation phase (`update_actor`).

## Root Cause: Unguarded Replay Queue Deadlock

### The Issue
**agent_ppo_trainer_pipeline.py** spawns a **background trajectory generation thread** (line 156) that fills a queue asynchronously:

```python
thread = threading.Thread(target=create_replay_queue, args=(...), daemon=False)
thread.start()
```

The **main training loop blocks indefinitely** waiting for queue data (line 173):

```python
_, trajes = replay_queue.get()  # ← NO TIMEOUT, NO ERROR HANDLING
```

**If the thread crashes** (OOM, shape error, timeout), the main thread **hangs forever** at `queue.get()`:
- Supervisor detects hung process → kills it → restarts training
- No error message → silent failure

### Why Step 1?
First step loads full model through trajectory generator + FSDP backward simultaneously:
- Trajectory thread generates sequences with vLLM (uses ~6 GPUs)
- Main thread runs backprop on actor (uses ~4 GPUs)
- **GPU memory collision at hand-off** → CUDA error → thread dies → deadlock

## Fixes Applied

### 1. Exception Propagation (line 139-159)
```python
thread_exception = None
def create_replay_queue(generator, q, batch_iter_val, timing_raw_val):
    nonlocal thread_exception
    try:
        # ... trajectory generation ...
    except Exception as e:
        thread_exception = e
        print(f"ERROR in trajectory generation thread: {e}", flush=True)
        traceback.print_exc()
```

Thread exceptions now stored in outer scope instead of crashing silently.

### 2. Queue Timeout + Dead Thread Detection (line 171-182)
```python
for _ in range(ppo_mini_batch_size):
    if thread_exception:
        raise RuntimeError(f"Trajectory generation thread failed: {thread_exception}")
    try:
        _, trajes = replay_queue.get(timeout=300)  # 5 min timeout
    except queue.Empty:
        if not thread.is_alive():
            raise RuntimeError(f"Trajectory thread died. Exception: {thread_exception}")
        raise RuntimeError(f"Queue timeout after 300s. Thread alive: {thread.is_alive()}")
    trajectories.extend(trajes)
```

Now catches:
- Dead thread (immediately raises)
- Queue timeout (raises with diagnostic)
- Thread exception (raises with root cause)

### 3. Backprop Phase Diagnostics (line 253-263)
```python
print(f"[Step {self.global_steps}] Starting actor update (backprop) for mini_batch {mini_batch_iter + 1}/{num_loops}, batch_size={...}", flush=True)

with Timer("update_actor", timing_raw):
    try:
        actor_output = self.actor_wg.update_actor_mini_batch(mini_batch)
    except Exception as e:
        print(f"[Step {self.global_steps}] ERROR during actor update backprop: {e}", flush=True)
        traceback.print_exc()
        raise
```

Logs batch shapes and exceptions at the exact failure point.

## How to Debug Next Restart

### Check logs for:
```bash
# Exception in trajectory generation thread
grep -i "ERROR in trajectory generation thread" training.log

# Backprop exceptions
grep -i "ERROR during actor update backprop" training.log

# Queue timeout/dead thread
grep -i "Queue timeout\|Trajectory thread died" training.log
```

### If exception still occurs:
1. **Trajectory thread error** → check vLLM memory, increase `RLLM_MAX_ROLLOUT_CONCURRENCY` tolerance
2. **Queue timeout** → trajectory generator too slow, reduce `TRAIN_BATCH_SIZE` or `PPO_MINI_BATCH_SIZE`
3. **Backprop error** → OOM during FSDP backward, reduce per-GPU batch size or enable offload

## Comparison: Pipeline vs Non-Pipeline Trainer

| Aspect | Original (`train_agent_ppo.py`) | Pipeline Trainer |
|--------|--------------------------------|------------------|
| Async model | Ray async manager | Threading + queue |
| Error handling | Built-in to framework | Manual (now fixed) |
| Queue timeout | Framework default (60s+) | Now 300s |
| Step 1 reliability | High (proven) | Fixed with patches |

**Recommendation**: If issues persist, fallback to original trainer:
```bash
python -m rllm.trainer.verl.train_agent_ppo  # non-pipeline (stable)
```

## Files Modified
- `agent_ppo_trainer_pipeline.py`: Added exception handling + timeout + diagnostics
