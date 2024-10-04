from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np
import torch as th
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.sac import SAC as OriginalSAC
from stable_baselines3.sac.policies import SACPolicy
from torch.nn import functional as F

from losses import biev_loss, biv_loss, gaussian_nll_loss, ggd_nll_loss, mse_loss

from ..common.algo import OffPolicyAlgorithm
from .policies import get_policy_aliases


class SAC(OffPolicyAlgorithm, OriginalSAC):
    def __init__(
        self,
        policy: Union[str, Type[SACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
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
            ent_coef=ent_coef,
            target_update_interval=target_update_interval,
            target_entropy=target_entropy,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
        )

    @property
    def policy_aliases(self) -> Dict[str, Type[SACPolicy]]:
        return get_policy_aliases(self.ensemble_size, self.distribution)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        actor_losses, ent_coefs, ent_coef_losses, critic_losses = [], [], [], []

        _params = []

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(  # type: ignore
                batch_size, env=self._vec_normalize_env
            )

            actions, log_prob = self.actor.action_log_prob(replay_data.observations)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = self.log_ent_coef.detach().exp()
                ent_coef_loss = -(
                    self.log_ent_coef
                    * (log_prob[..., None] + self.target_entropy).detach()
                ).mean()
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()
            else:
                ent_coef = self.ent_coef_tensor

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(
                    replay_data.next_observations
                )
                next_values = self.critic_target(
                    replay_data.next_observations, next_actions
                )
                next_values, _ = next_values[..., 0].min(dim=0)
                next_values = next_values - ent_coef * next_log_prob[..., None]
                target_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * self.gamma * next_values
                )

            current_values = self.critic(replay_data.observations, replay_data.actions)

            critic_loss = 0.5 * sum(
                self.value_loss_fn(_current_values, target_values)
                for _current_values in current_values
            )
            assert isinstance(critic_loss, th.Tensor)

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            action_values = self.critic(replay_data.observations, actions)
            action_values, _ = action_values[..., 0].mean(dim=2).min(dim=0)
            actor_loss = (ent_coef * log_prob - action_values).mean()

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(
                    self.critic.parameters(), self.critic_target.parameters(), self.tau
                )
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1)

            critic_losses.append(critic_loss.item())
            actor_losses.append(actor_loss.item())
            ent_coefs.append(ent_coef.item())
            if ent_coef_loss is not None:
                ent_coef_losses.append(ent_coef_loss.item())

            _params.append(F.softplus(current_values[..., -1]).detach())

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

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


class GD_SAC(SAC):
    distribution = "gaussian"

    def value_loss_fn(
        self, current_values: th.Tensor, target_values: th.Tensor
    ) -> th.Tensor:
        return gaussian_nll_loss(current_values, target_values)


class IV_GD_SAC(GD_SAC):
    def value_loss_fn(
        self, current_values: th.Tensor, target_values: th.Tensor
    ) -> th.Tensor:
        return gaussian_nll_loss(
            current_values, target_values
        ) + self.uncertainty_temperature * biv_loss(
            current_values, target_values, self.gamma, self.min_batch_size, "mse"
        )


class GGD_SAC(SAC):
    distribution = "ggd"

    def value_loss_fn(
        self, current_values: th.Tensor, target_values: th.Tensor
    ) -> th.Tensor:
        return ggd_nll_loss(current_values, target_values)


class IV_GGD_SAC(GGD_SAC):
    def value_loss_fn(
        self, current_values: th.Tensor, target_values: th.Tensor
    ) -> th.Tensor:
        return ggd_nll_loss(
            current_values, target_values
        ) + self.uncertainty_temperature * biv_loss(
            current_values, target_values, self.gamma, self.min_batch_size, "mae"
        )


class IEV_GGD_SAC(GGD_SAC):
    def value_loss_fn(
        self, current_values: th.Tensor, target_values: th.Tensor
    ) -> th.Tensor:
        return ggd_nll_loss(
            current_values, target_values
        ) + self.uncertainty_temperature * biev_loss(
            current_values, target_values, self.min_batch_size, "mae"
        )
