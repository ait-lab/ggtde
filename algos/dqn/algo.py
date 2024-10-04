from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np
import torch as th
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.dqn import DQN as OriginalDQN
from stable_baselines3.dqn.policies import DQNPolicy
from torch import nn
from torch.nn import functional as F

from losses import biev_loss, biv_loss, gaussian_nll_loss, ggd_nll_loss, mse_loss

from ..common.algo import OffPolicyAlgorithm
from .policies import get_policy_aliases


class DQN(OffPolicyAlgorithm, OriginalDQN):
    def __init__(
        self,
        policy: Union[str, Type[DQNPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 50_000,
        batch_size: int = 32,
        tau: float = 1,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        target_update_interval: int = 10_000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        ensemble_size: int = 5,
        min_batch_size: int = 16,
        uncertainty_temperature: float = 0.1,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 1,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        **kwargs,
    ):
        self.ensemble_size = ensemble_size
        self.value_dims = 2 - int(self.distribution is None)
        self.min_batch_size = min_batch_size
        self.uncertainty_temperature = uncertainty_temperature

        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            max_grad_norm=max_grad_norm,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
        )

    @property
    def policy_aliases(self) -> Dict[str, Type[DQNPolicy]]:
        return get_policy_aliases(self.ensemble_size, self.distribution)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []

        _params = []

        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(  # type: ignore
                batch_size, env=self._vec_normalize_env
            )

            with th.no_grad():
                next_values = self.q_net_target(replay_data.next_observations)
                next_values, _ = next_values[..., 0].max(dim=2)
                target_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * self.gamma * next_values
                )

            current_values = self.q_net(replay_data.observations)
            actions = (
                replay_data.actions[..., None, None]
                .long()
                .expand(-1, self.ensemble_size, 1, self.value_dims)
            )
            current_values = current_values.gather(dim=2, index=actions)[..., 0, :]

            loss = self.value_loss_fn(current_values, target_values)

            self.policy.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            losses.append(loss.item())

            _params.append(F.softplus(current_values[..., -1]).detach())

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

        if self.distribution:
            if self.distribution == "gaussian":
                param = "var"
            elif self.distribution == "ggd":
                param = "beta"
            _params = th.concat(_params)
            self.logger.record(f"v_approx/avg_{param}", _params.mean().item())
            self.logger.record(f"v_approx/std_{param}", _params.std().item())
            self.logger.record(f"v_approx/max_{param}", _params.max().item())
            self.logger.record(f"v_approx/min_{param}", _params.min().item())

    def value_loss_fn(
        self, current_values: th.Tensor, target_values: th.Tensor
    ) -> th.Tensor:
        return mse_loss(current_values, target_values)


class GD_DQN(DQN):
    distribution = "gaussian"

    def value_loss_fn(
        self, current_values: th.Tensor, target_values: th.Tensor
    ) -> th.Tensor:
        return gaussian_nll_loss(current_values, target_values)


class IV_GD_DQN(GD_DQN):
    def value_loss_fn(
        self, current_values: th.Tensor, target_values: th.Tensor
    ) -> th.Tensor:
        return gaussian_nll_loss(
            current_values, target_values
        ) + self.uncertainty_temperature * biv_loss(
            current_values, target_values, self.gamma, self.min_batch_size, "mse"
        )


class GGD_DQN(DQN):
    distribution = "ggd"

    def value_loss_fn(
        self, current_values: th.Tensor, target_values: th.Tensor
    ) -> th.Tensor:
        return ggd_nll_loss(current_values, target_values)


class IV_GGD_DQN(GGD_DQN):
    def value_loss_fn(
        self, current_values: th.Tensor, target_values: th.Tensor
    ) -> th.Tensor:
        return ggd_nll_loss(
            current_values, target_values
        ) + self.uncertainty_temperature * biv_loss(
            current_values, target_values, self.gamma, self.min_batch_size, "mae"
        )


class IEV_GGD_DQN(GGD_DQN):
    def value_loss_fn(
        self, current_values: th.Tensor, target_values: th.Tensor
    ) -> th.Tensor:
        return ggd_nll_loss(
            current_values, target_values
        ) + self.uncertainty_temperature * biev_loss(
            current_values, target_values, self.min_batch_size, "mae"
        )
