import queue
import threading
import time
import uuid
from pprint import pprint
from queue import Queue

import numpy as np
import torch

from rllm.engine.agent_execution_engine import AsyncAgentExecutionEngine
from rllm.trainer.verl.agent_ppo_trainer import AgentPPOTrainer
from verl import DataProto
from verl.single_controller.ray import (
    RayClassWithInitArgs,
    RayWorkerGroup,
)
from verl.trainer.ppo.ray_trainer import (
    Role,
    compute_advantage,
    compute_data_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer_pipeline import (
    Timer,
    update_metrics,
)


class PipelineAgentPPOTrainer(AgentPPOTrainer):
    def init_workers(self):
        assert self.config.agent.async_engine, "PPO pipeline trainer must use async engine"
        assert not self.hybrid_engine, "PPO pipeline trainer does not support hybrid engine, assumes Rollout and Actor are not in the different worker group"
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        assert Role.Actor in self.role_worker_mapping and Role.Rollout in self.role_worker_mapping, "Actor and Rollout must be in role_worker_mapping"
        actor_resource_pool = self.resource_pool_manager.get_resource_pool(Role.Actor)
        # actor_gpu_ids = actor_resource_pool.gpu_assignments if isinstance(actor_resource_pool, RayResourcePool) else None

        actor_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.Actor],
            config=self.config.actor_rollout_ref,
            role="actor",
            reward_config=self.config.reward_model,
        )
        self.resource_pool_to_cls[actor_resource_pool]["actor"] = actor_cls

        # Get rollout resource pool
        rollout_resource_pool = self.resource_pool_manager.get_resource_pool(Role.Rollout)
        # rollout_gpu_ids = rollout_resource_pool.gpu_assignments if isinstance(rollout_resource_pool, RayResourcePool) else None
        rollout_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.Rollout],
            config=self.config.actor_rollout_ref,
            role="rollout",
            reward_config=self.config.reward_model,
        )
        self.resource_pool_to_cls[rollout_resource_pool]["rollout"] = rollout_cls

        self.actor_wg = RayWorkerGroup(resource_pool=actor_resource_pool, ray_cls_with_init=actor_cls)
        self.rollout_wg = RayWorkerGroup(resource_pool=rollout_resource_pool, ray_cls_with_init=rollout_cls)

        self.actor_wg.init_model()
        self.rollout_wg.init_model()
        self.rollout_wg.tp_size = self.config.actor_rollout_ref.rollout.get("tensor_model_parallel_size", 1)

        self.agent_execution_engine = AsyncAgentExecutionEngine(
            rollout_engine=self.rollout_wg,
            config=self.config,
            engine_name="verl",
            tokenizer=self.tokenizer,
            model_path=self.config.actor_rollout_ref.model.path,
            max_steps=self.config.agent.max_steps,
            max_response_length=self.config.data.max_response_length,
            max_prompt_length=self.config.data.max_prompt_length,
            agent_class=self.agent_class,
            agent_args=self.config.agent.get("agent_args", {}),
            env_class=self.env_class,
            env_args=self.config.env.get("env_args", {}),
            **self.config.agent.get("engine_args", {}),
        )

    def fit_agent(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(project_name=self.config.trainer.project_name, experiment_name=self.config.trainer.experiment_name, default_backend=self.config.trainer.logger, config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate_agent()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # we start from step 1
        self.global_steps += 1

        replay_queue = Queue()
        total_mini_batch_iters = 0
        for epoch in range(self.config.trainer.total_epochs):
            for batch_iter, batch_dict in enumerate(self.train_dataloader):
                print(f"step: {self.global_steps}", flush=True)

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                batch = batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n,
                    interleave=True,
                )

                # must pop those keys for generation so they no longer exist
                batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"])
                batch.meta_info = {
                    "agent_rollout": True,  # no need to generate multiple ones since environment is repeated already
                }

                metrics = {}
                timing_raw = {}

                self.init_envs_and_agents(batch)

                with Timer("step", timing_raw):

                    thread_exception = [None]  # mutable container to capture exception in thread
                    def create_replay_queue(generator, q, batch_iter_val, timing_raw_val):
                        try:
                            uid_to_trajectories = {}
                            with Timer("gen", timing_raw_val):
                                for _, trajectory in enumerate(generator):
                                    uid = trajectory["idx"] // self.config.actor_rollout_ref.rollout.n
                                    if uid not in uid_to_trajectories:
                                        uid_to_trajectories[uid] = []
                                    uid_to_trajectories[uid].append(trajectory)
                                    if len(uid_to_trajectories[uid]) == self.config.actor_rollout_ref.rollout.n:
                                        q.put((batch_iter_val, uid_to_trajectories[uid]))
                                        del uid_to_trajectories[uid]
                        except Exception as e:
                            thread_exception[0] = e
                            print(f"\n[ERROR] Trajectory generation thread crashed:", flush=True)
                            print(f"  {type(e).__name__}: {e}", flush=True)
                            import traceback
                            traceback.print_exc()

                    if self.config.agent.step_advantage_broadcast:
                        raise Exception("Stepwise advantage broadcasting not supported on pipelined trainer yet")
                    gen_seq_generator = self.generate_agent_trajectories_async(timing_raw=timing_raw, meta_info=batch.meta_info)
                    thread = threading.Thread(target=create_replay_queue, args=(gen_seq_generator, replay_queue, batch_iter, timing_raw), daemon=False)
                    thread.start()

                    ppo_train_batch_size = self.config.data.train_batch_size
                    ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
                    assert ppo_train_batch_size % ppo_mini_batch_size == 0, "PPO mini batch size must be a divisor of the total training batch size"
                    ppo_step_minibatch_iter = ppo_train_batch_size // ppo_mini_batch_size
                    num_loops = ppo_step_minibatch_iter
                    training_batch = []
                    for mini_batch_iter in range(num_loops):
                        print(f"mini_batch_iter: {mini_batch_iter + 1} / {num_loops}", flush=True)
                        mini_batch_metrics = {}
                        start_time = time.perf_counter()
                        with Timer("pipeline_gen", timing_raw):
                            trajectories = []
                            try:
                                for i in range(ppo_mini_batch_size):
                                    if thread_exception[0]:
                                        raise RuntimeError(f"Trajectory thread failed: {thread_exception[0]}")
                                    try:
                                        _, trajes = replay_queue.get(timeout=300)  # 5min timeout
                                        trajectories.extend(trajes)
                                    except queue.Empty:
                                        if not thread.is_alive():
                                            raise RuntimeError(f"Trajectory thread died. Exception: {thread_exception[0]}")
                                        raise RuntimeError(f"Queue timeout {300}s (thread alive but stuck)")
                            except RuntimeError as e:
                                print(f"\n[BATCH SKIP] Trajectory generation failed, skipping batch:", flush=True)
                                print(f"  {e}", flush=True)
                                break  # Skip this batch
                            mini_batch = self._transform_agent_trajectories(trajectories=trajectories)
                            ids2uids = [traj["idx"] // self.config.actor_rollout_ref.rollout.n for traj in trajectories]
                            mini_batch.non_tensor_batch["uid"] = np.array(ids2uids, dtype=object)
                        end_time = time.perf_counter()
                        print(f"Generate mini batch took {end_time - start_time:.2f} seconds")

                        if total_mini_batch_iters % ppo_step_minibatch_iter == ppo_step_minibatch_iter - 1:
                            mini_batch.meta_info["last_mini_batch"] = True

                        with Timer("adv", timing_raw):
                            reward_tensor = mini_batch.batch["token_level_scores"]  # already computed
                            print("Reward tensor:", reward_tensor.sum(-1))

                            # Group rewards by uid
                            uids = mini_batch.non_tensor_batch["uid"]
                            unique_uids = np.unique(uids)
                            valid_mask = torch.ones(len(uids), dtype=torch.bool)
                            solve_none = 0
                            solve_all = 0
                            solve_partial = 0

                            print(f"num unique_uids: {len(unique_uids)}")
                            for uid in unique_uids:
                                uid_mask = uids == uid
                                uid_rewards = reward_tensor[uid_mask].sum(-1)  # Sum rewards for each sequence

                                # Check if all rewards are 0 or all are 1 for this uid
                                if (uid_rewards == 0).all():
                                    valid_mask[uid_mask] = False
                                    solve_none += 1
                                elif (uid_rewards == 1).all():
                                    valid_mask[uid_mask] = False
                                    solve_all += 1
                                else:
                                    solve_partial += 1

                            # Log to metrics
                            mini_batch_metrics["batch/solve_none"] = solve_none
                            mini_batch_metrics["batch/solve_all"] = solve_all
                            mini_batch_metrics["batch/solve_partial"] = solve_partial

                            # Recompute old_log_probs using Pytorch FSDP.
                            with Timer("old_log_prob", timing_raw):
                                old_log_prob = self.actor_wg.compute_log_prob(mini_batch)
                                mini_batch = mini_batch.union(old_log_prob)

                            if self.use_reference_policy:
                                # compute reference log_prob
                                with Timer("ref", timing_raw):
                                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(mini_batch)
                                    mini_batch = mini_batch.union(ref_log_prob)

                            mini_batch.batch["token_level_rewards"] = mini_batch.batch["token_level_scores"]
                            # compute advantages, executed on the driver process
                            mini_batch = compute_advantage(mini_batch, adv_estimator=self.config.algorithm.adv_estimator, gamma=self.config.algorithm.gamma, lam=self.config.algorithm.lam, mask_truncated_samples=self.config.algorithm.mask_truncated_samples, clip_advantages=self.config.algorithm.clip_advantages)

                        self._balance_batch(mini_batch, metrics=mini_batch_metrics)
                        mini_batch.meta_info["global_token_num"] = torch.sum(mini_batch.batch["attention_mask"], dim=-1).tolist()
                        # update actor (backpropagation — most likely crash point)
                        start_time = time.perf_counter()
                        print(f"[Step {self.global_steps}] Actor backprop: batch_size={mini_batch.batch['input_ids'].shape[0]}", flush=True)

                        try:
                            with Timer("update_actor", timing_raw):
                                actor_output = self.actor_wg.update_actor_mini_batch(mini_batch)
                            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                            end_time = time.perf_counter()
                            print(f"[Step {self.global_steps}] Actor update OK ({end_time - start_time:.2f}s)", flush=True)
                            mini_batch_metrics.update(actor_output_metrics)
                            training_batch.append(mini_batch)
                            update_metrics(metrics, mini_batch_metrics)
                            total_mini_batch_iters += 1
                        except Exception as e:
                            print(f"\n[ERROR] Actor backprop failed:", flush=True)
                            print(f"  {type(e).__name__}: {e}", flush=True)
                            import traceback
                            traceback.print_exc()
                            print(f"[SKIP] Skipping this mini-batch and continuing to next\n", flush=True)

                    # last_iter_mini_batch_iter = (mini_batch_iter + last_iter_mini_batch_iter - 1) % ppo_step_minibatch_iter
                    with Timer("rollout_model_update", timing_raw):
                        updated_actor_module_fsdp_ref = self.actor_wg.get_state_dict()
                        if isinstance(updated_actor_module_fsdp_ref, list):
                            updated_actor_module_fsdp_ref = updated_actor_module_fsdp_ref[0]
                        self.rollout_wg.update_rollout_actor_module(updated_actor_module_fsdp_ref)
                    training_batch = DataProto.concat(training_batch)

                    # Validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and self.global_steps % self.config.trainer.test_freq == 0:
                        with Timer("testing", timing_raw):
                            val_metrics: dict = self._validate_agent()
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and self.global_steps % self.config.trainer.save_freq == 0:
                        with Timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                metrics.update(compute_data_metrics(batch=training_batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=training_batch, timing_raw=timing_raw))

                for k, v in metrics.items():
                    if isinstance(v, list | np.ndarray):
                        if "batch/" in k:
                            metrics[k] = np.sum(v)
                        else:
                            metrics[k] = np.mean(v)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)
                self.global_steps += 1

                if self.val_reward_fn is not None and self.global_steps >= self.total_training_steps:
                    # perform validation after training
                    val_metrics = self._validate_agent()
                    pprint(f"Final validation metrics: {val_metrics}")
                    logger.log(data=val_metrics, step=self.global_steps)
                    if self.config.trainer.save_freq > 0 and (self.global_steps - 1) % self.config.trainer.save_freq != 0:
                        with Timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()
                    return

    def _validate_agent(self):
        rewards_lst = []
        env_rewards_lst = []
        data_source_lst = []
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            n_val_samples = self.config.actor_rollout_ref.rollout.val_kwargs.n
            test_batch = test_batch.repeat(repeat_times=n_val_samples, interleave=True)
            test_batch.pop(["input_ids", "attention_mask", "position_ids"])  # these are not needed for environment based interaction
            test_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "agent_rollout": True,
            }

            self.init_envs_and_agents(test_batch)

            test_traj_generator = self.generate_agent_trajectories_async(meta_info=test_batch.meta_info)

            trajectories = []
            for traj in test_traj_generator:
                trajectories.append(traj)
            test_output_gen_batch = self._transform_agent_trajectories(trajectories=trajectories)

            if test_batch.meta_info["recompute_log_prob"]:
                with torch.no_grad():
                    output = self.actor_wg.compute_log_prob(test_output_gen_batch)
                    test_output_gen_batch = test_output_gen_batch.union(output)

            test_batch = test_batch.union(test_output_gen_batch)

            # use environment score to report validation reward
            reward_tensor = test_batch.batch["token_level_scores"]
            env_reward_tensor = test_batch.batch["environment_scores"]

            rewards_lst.append(reward_tensor.sum(-1).cpu())
            env_rewards_lst.append(env_reward_tensor.sum(-1).cpu())
            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        reward_tensor = torch.cat(rewards_lst, dim=0)  # (batch_size,)
        env_reward_tensor = torch.cat(env_rewards_lst, dim=0)  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        # evaluate test_score based on data source
        data_source_reward = {}
        data_source_env_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]

            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

            if data_source not in data_source_env_reward:
                data_source_env_reward[data_source] = []
            data_source_env_reward[data_source].append(env_reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f"val/test_score/{data_source}"] = np.mean(rewards)

        for data_source, env_rewards in data_source_env_reward.items():
            metric_dict[f"val/env_score/{data_source}"] = np.mean(env_rewards)

        return metric_dict
