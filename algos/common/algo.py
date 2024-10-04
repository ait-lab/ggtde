import sys
import time
from typing import Literal, Optional

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import (
    OffPolicyAlgorithm as OriginalOffPolicyAlgorithm,
)
from stable_baselines3.common.on_policy_algorithm import (
    OnPolicyAlgorithm as OriginalOnPolicyAlgorithm,
)
from stable_baselines3.common.type_aliases import (
    MaybeCallback,
    RolloutReturn,
    TrainFreq,
    TrainFrequencyUnit,
)
from stable_baselines3.common.utils import (
    obs_as_tensor,
    safe_mean,
    should_collect_more_steps,
)
from stable_baselines3.common.vec_env import VecEnv

from utils.buffers import RolloutBuffer


class OnPolicyAlgorithm(OriginalOnPolicyAlgorithm):
    distribution: Optional[Literal["gaussian", "ggd"]] = None

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        reset_num_timesteps: bool = True,
        **kwargs,
    ) -> "OnPolicyAlgorithm":
        iteration = 0
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback=callback,
            reset_num_timesteps=reset_num_timesteps,
            tb_log_name=self.__class__.__name__,
        )

        callback.on_training_start(locals(), globals())

        assert (
            self.env is not None
        ), "You must set the environment before calling learn()"

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(
                self.env,
                callback,
                self.rollout_buffer,  # type: ignore
                self.n_steps,
            )

            if not continue_training:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                time_elapsed = max(
                    (time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon
                )
                fps = int(
                    (self.num_timesteps - self._num_timesteps_at_start) / time_elapsed
                )
                self.logger.record("time/iteration", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record(
                        "rollout/avg_reward",
                        safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]),
                    )
                    self.logger.record(
                        "rollout/avg_epilen",
                        safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]),
                    )
                self.logger.record("time/fps", fps)
                self.logger.record(
                    "time/time", int(time_elapsed), exclude="tensorboard"
                )
                self.logger.dump(step=self.num_timesteps)

            self.train()

        callback.on_training_end()
        return self

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)  # type: ignore
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    clipped_actions = np.clip(
                        actions, self.action_space.low, self.action_space.high
                    )

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(
                        infos[idx]["terminal_observation"]
                    )[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[
                            0
                        ].mean(dim=0)
                    rewards[idx] += self.gamma * terminal_value[0]

            rollout_buffer.add(
                self._last_obs,  # type: ignore
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore
                values,
                log_probs,
            )
            self._last_obs = new_obs  # type: ignore
            self._last_episode_starts = dones

        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore

        rollout_buffer.compute_returns_and_advantage(values, dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True


class OffPolicyAlgorithm(OriginalOffPolicyAlgorithm):
    distribution: Optional[Literal["gaussian", "ggd"]] = None

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        reset_num_timesteps: bool = True,
        **kwargs,
    ) -> "OffPolicyAlgorithm":
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback=callback,
            reset_num_timesteps=reset_num_timesteps,
            tb_log_name=self.__class__.__name__,
        )

        callback.on_training_start(locals(), globals())

        assert (
            self.env is not None
        ), "You must set the environment before calling learn()"
        assert isinstance(self.train_freq, TrainFreq)

        while self.num_timesteps < total_timesteps:
            rollout = self.collect_rollouts(
                self.env,
                callback,
                self.train_freq,
                self.replay_buffer,  # type: ignore
                action_noise=self.action_noise,
                learning_starts=self.learning_starts,
                log_interval=log_interval,
            )

            if not rollout.continue_training:
                break

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                gradient_steps = (
                    self.gradient_steps
                    if self.gradient_steps >= 0
                    else rollout.episode_timesteps
                )
                if gradient_steps > 0:
                    self.train(gradient_steps, self.batch_size)

        callback.on_training_end()
        return self

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."
        if env.num_envs > 1:
            assert (
                train_freq.unit == TrainFrequencyUnit.STEP
            ), "You must use only one env when doing episodic training."
        self.policy.set_training_mode(False)

        n_steps, n_episodes = 0, 0

        callback.on_rollout_start()

        continue_training = True
        while should_collect_more_steps(train_freq, n_steps, n_episodes):
            actions, buffer_actions = self._sample_action(
                learning_starts, action_noise=action_noise, n_envs=env.num_envs
            )

            new_obs, rewards, dones, infos = env.step(actions)

            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if not callback.on_step():
                return RolloutReturn(
                    episode_timesteps=n_steps * env.num_envs,
                    n_episodes=n_episodes,
                    continue_training=False,
                )

            self._update_info_buffer(infos, dones=dones)
            n_steps += 1

            self._store_transition(
                replay_buffer, buffer_actions, new_obs, rewards, dones, infos  # type: ignore
            )

            self._update_current_progress_remaining(
                self.num_timesteps, self._total_timesteps
            )

            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    n_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    if (
                        log_interval is not None
                        and self._episode_num % log_interval == 0
                    ):
                        self._dump_logs()

        callback.on_rollout_end()

        return RolloutReturn(
            episode_timesteps=n_steps * env.num_envs,
            n_episodes=n_episodes,
            continue_training=continue_training,
        )
