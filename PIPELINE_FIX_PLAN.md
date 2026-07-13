# Pipeline Trainer Step-1 Fix Plan: Robust Error Recovery

## Current Failure Mode
```
Step 1: Thread spawns → generates trajectories (vLLM on 6 GPUs)
        Main thread starts backprop (FSDP on 4 GPUs)
        GPU collision → CUDA error in thread
        Thread dies silently
        Main thread blocks forever at queue.get()
        Supervisor kills hung process → RESTART
```

## Root Problems
1. **No pre-flight check** — errors caught too late (during backprop)
2. **Silent thread death** — exception lost, queue deadlocks
3. **No recovery mechanism** — one thread failure = entire run dies
4. **Threading fragile** — Ray's async manager is more robust

---

## Solution Architecture: 3-Phase Approach

### Phase 1: Early Error Detection (Pre-Training Warmup)
**Goal**: Catch vLLM/CUDA errors BEFORE step 1 backprop

**Implementation**:
```python
def _warmup_trajectory_generation(self):
    """Pre-generate N trajectories to validate pipeline before training loop."""
    # Get single batch from dataloader
    batch_dict = next(iter(self.train_dataloader))
    batch = DataProto.from_single_dict(batch_dict)
    
    # Repeat small batch (n=2-4 samples only, not full ppo_mini_batch_size)
    batch = batch.repeat(repeat_times=2, interleave=True)
    batch.pop(["input_ids", "attention_mask", "position_ids"])
    batch.meta_info = {"agent_rollout": True}
    
    self.init_envs_and_agents(batch)
    
    # Try to generate trajectories synchronously — will fail fast if vLLM broken
    try:
        print("[WARMUP] Attempting synchronous trajectory generation...")
        gen = self.generate_agent_trajectories_async(timing_raw={}, meta_info=batch.meta_info)
        trajs = []
        for i, traj in enumerate(gen):
            trajs.append(traj)
            if i >= 5:  # Just grab first 6 to test
                break
        print(f"[WARMUP] ✓ Generated {len(trajs)} test trajectories successfully")
        return trajs
    except Exception as e:
        print(f"[WARMUP] ✗ Trajectory generation failed: {e}")
        raise
```

**Outcome**: 
- ✅ If warmup passes → proceed to async threaded pipeline (now validated)
- ❌ If warmup fails → fall back to synchronous trajectory generation (slower but works)

---

### Phase 2: Thread Watchdog + Auto-Restart
**Goal**: Detect dead threads, restart them automatically

**Implementation**:
```python
class TrajectoryThreadPool:
    """Manages trajectory generation threads with automatic restart."""
    
    def __init__(self, config):
        self.thread = None
        self.queue = Queue()
        self.exception = None
        self.max_retries = 3
        self.retry_count = 0
        self.config = config
    
    def start_thread(self, generator, batch_iter):
        """Start trajectory generation thread with exception handling."""
        self.exception = None
        self.thread = threading.Thread(
            target=self._run_with_exception_handler,
            args=(generator, batch_iter),
            daemon=False
        )
        self.thread.start()
    
    def _run_with_exception_handler(self, generator, batch_iter):
        """Wrap thread execution with proper exception capture."""
        try:
            uid_to_trajectories = {}
            for _, trajectory in enumerate(generator):
                uid = trajectory["idx"] // self.config.actor_rollout_ref.rollout.n
                if uid not in uid_to_trajectories:
                    uid_to_trajectories[uid] = []
                uid_to_trajectories[uid].append(trajectory)
                if len(uid_to_trajectories[uid]) == self.config.actor_rollout_ref.rollout.n:
                    self.queue.put((batch_iter, uid_to_trajectories[uid]))
                    del uid_to_trajectories[uid]
            print("[TrajectoryThread] ✓ Trajectory generation completed", flush=True)
        except Exception as e:
            self.exception = e
            print(f"[TrajectoryThread] ✗ Exception: {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc()
    
    def get_with_retry(self, timeout=300):
        """Get from queue with automatic thread restart on failure."""
        while True:
            # Check if thread is dead
            if not self.thread.is_alive():
                if self.exception:
                    raise RuntimeError(f"Trajectory thread died: {self.exception}")
                else:
                    raise RuntimeError("Trajectory thread died unexpectedly")
            
            try:
                return self.queue.get(timeout=timeout)
            except queue.Empty:
                # Queue timeout — thread might be stuck
                if not self.thread.is_alive():
                    raise RuntimeError(f"Trajectory thread timeout then died: {self.exception}")
                raise RuntimeError(f"Queue timeout after {timeout}s (thread still alive, but not producing)")
```

**Outcome**:
- Thread crash detected immediately at next `get_with_retry()` call
- Raises descriptive error (not silent deadlock)
- Caller can catch + retry

---

### Phase 3: Graceful Degradation (Sync Fallback)
**Goal**: If async fails, don't crash — fall back to slower sync mode

**Implementation**:
```python
def _get_trajectory_generator(self, use_sync=False):
    """Return trajectory generator (async or sync based on use_sync flag)."""
    if use_sync:
        print("[TrajectoryGen] Using SYNCHRONOUS trajectory generation (slower but robust)", flush=True)
        return self.generate_agent_trajectories_sync(meta_info=self.batch.meta_info)
    else:
        print("[TrajectoryGen] Using ASYNCHRONOUS trajectory generation (faster)", flush=True)
        return self.generate_agent_trajectories_async(meta_info=self.batch.meta_info)

# In fit_agent loop:
try:
    gen = self._get_trajectory_generator(use_sync=False)
    thread_pool.start_thread(gen, batch_iter)
    # ... proceed with async queue.get() ...
except (RuntimeError, queue.Empty) as e:
    print(f"[ERROR] Async trajectory generation failed: {e}", flush=True)
    print("[FALLBACK] Switching to synchronous mode and retrying batch...", flush=True)
    
    # Reinitialize batch and try sync generation
    self.init_envs_and_agents(batch)
    gen = self._get_trajectory_generator(use_sync=True)
    
    # Generate trajectories synchronously (blocking, but error is immediate)
    trajectories = []
    for traj in gen:
        trajectories.extend([traj] if isinstance(traj, dict) else traj)
    
    # Process batch as normal
    mini_batch = self._transform_agent_trajectories(trajectories)
    # ... continue training with this batch ...
```

**Outcome**:
- Async fails → automatic fallback to sync (slower but always works)
- No silent hangs, no restarts
- Training continues (just slower for this batch/epoch)

---

## Implementation Order (Phases)

### Phase 1: Warmup (15 min)
- Add `_warmup_trajectory_generation()` method
- Call at start of `fit_agent()` before epoch loop
- File: `agent_ppo_trainer_pipeline.py`

### Phase 2: Thread Watchdog (30 min)
- Create `TrajectoryThreadPool` class
- Replace bare `threading.Thread` with pool
- Replace `replay_queue.get()` with `pool.get_with_retry()`
- File: `agent_ppo_trainer_pipeline.py` (or extract to new file)

### Phase 3: Sync Fallback (20 min)
- Add `generate_agent_trajectories_sync()` method (or check if exists)
- Wrap async thread spawn in try-except
- On exception, fall back to sync generation
- File: `agent_ppo_trainer_pipeline.py`

---

## Success Criteria

| Scenario | Before | After |
|----------|--------|-------|
| vLLM CUDA error on step 1 | Silent hang → restart | Caught at warmup → fall back to sync |
| Thread crashes mid-training | Main thread hangs | Detected + raises descriptive error |
| Queue timeout | Supervisor kills process | Error logged, fallback to sync, training continues |
| Successful async trajectory gen | ✓ Works | ✓ Still works (no change) |

---

## Files to Modify

1. **agent_ppo_trainer_pipeline.py**
   - Add `_warmup_trajectory_generation()`
   - Add `TrajectoryThreadPool` class
   - Replace thread + queue logic with pool
   - Add try-except for async → sync fallback
   - Add `use_sync_trajectory_generation` flag to trainer

2. **agent_ppo_trainer.py** (parent class)
   - Add `generate_agent_trajectories_sync()` method if missing
   - Ensure it's compatible with async version (same output format)

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Warmup might be too fast to catch rare errors | Add `num_warmup_samples=10` param (configurable) |
| Sync fallback significantly slower | Accept slower speed vs. complete failure; log warning |
| Sync gen incompatible with async format | Test both return dict shapes before production |
| Extra logging slows down training | Gate logging behind `debug=True` flag |

---

## Recommended: Start with Phase 1 + 2
- Phase 1 catches errors early (fast to implement)
- Phase 2 detects thread failures (reliable)
- Phase 3 only needed if sync not available

**Estimated total effort**: ~45 min for all 3 phases
**Payoff**: Step-1 restarts → robust error messages + automatic recovery
