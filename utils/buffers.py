from typing import Generator, Optional, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer as OriginalRolloutBuffer
from stable_baselines3.common.type_aliases import RolloutBufferSamples
from stable_baselines3.common.vec_env import VecNormalize


class RolloutBuffer(OriginalRolloutBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
        ensemble_size: int = 5,
        value_dims: int = 1,
    ):
        self.ensemble_size = ensemble_size
        self.value_dims = value_dims

        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device=device,
            gae_lambda=gae_lambda,
            gamma=gamma,
            n_envs=n_envs,
        )

    def reset(self) -> None:
        super().reset()

        self.returns = np.zeros(
            (self.buffer_size, self.n_envs, self.ensemble_size),
            dtype=np.float32,
        )
        self.values = np.zeros(
            (self.buffer_size, self.n_envs, self.ensemble_size, self.value_dims),
            dtype=np.float32,
        )
        self.advantages = np.zeros(
            (self.buffer_size, self.n_envs, self.ensemble_size),
            dtype=np.float32,
        )

    def compute_returns_and_advantage(
        self, last_values: th.Tensor, dones: np.ndarray
    ) -> None:
        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1 - dones
                next_values = (
                    last_values.detach()
                    .cpu()
                    .numpy()
                    .reshape(-1, self.ensemble_size, self.value_dims)
                )
            else:
                next_non_terminal = 1 - self.episode_starts[step + 1]
                next_values = self.values[step + 1, ...]
            delta = (
                self.rewards[step, ..., None]
                + self.gamma * next_values[..., 0] * next_non_terminal[..., None]
                - self.values[step, ..., 0]
            )
            last_gae_lam = (
                delta
                + self.gamma
                * self.gae_lambda
                * next_non_terminal[..., None]
                * last_gae_lam
            )
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + self.values[..., 0]

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
    ) -> None:
        if len(log_prob.shape) == 0:
            log_prob = log_prob.reshape(-1, 1)

        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))

        action = action.reshape((self.n_envs, self.action_dim))

        self.observations[self.pos] = np.array(obs)
        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.episode_starts[self.pos] = np.array(episode_start)
        self.values[self.pos] = (
            value.detach()
            .cpu()
            .numpy()
            .reshape(-1, self.ensemble_size, self.value_dims)
        )
        self.log_probs[self.pos] = log_prob.detach().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(
        self, batch_size: Optional[int] = None
    ) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> RolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].reshape(-1, self.ensemble_size, self.value_dims),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].reshape(-1, self.ensemble_size),
            self.returns[batch_inds].reshape(-1, self.ensemble_size),
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))
