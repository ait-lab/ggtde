from typing import Any, Dict, Optional, Type, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.ppo import PPO as OriginalPPO
from torch import nn
from torch.nn import functional as F

from losses import biev_loss, biv_loss, gaussian_nll_loss, ggd_nll_loss, mse_loss
from utils.buffers import RolloutBuffer
from utils.constants import EPS

from ..common.algo import OnPolicyAlgorithm
from .policies import get_policy_aliases


class PPO(OnPolicyAlgorithm, OriginalPPO):
    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2_048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Optional[Union[float, Schedule]] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
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
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            rollout_buffer_class=RolloutBuffer,
            rollout_buffer_kwargs={
                "ensemble_size": ensemble_size,
                "value_dims": self.value_dims,
            },
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
        )

    @property
    def policy_aliases(self) -> Dict[str, Type[ActorCriticPolicy]]:
        return get_policy_aliases(self.ensemble_size, self.distribution)

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore

        pi_losses, ent_losses, clip_fractions, v_losses = [], [], [], []

        _params = []

        continue_training = True
        for _ in range(self.n_epochs):
            approx_kl_divs = []

            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )

                advantages = rollout_data.advantages.mean(dim=-1)
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + EPS
                    )

                ratio = (log_prob - rollout_data.old_log_prob).exp()
                pi_loss = -th.min(
                    advantages * ratio,
                    advantages * ratio.clip(1 - clip_range, 1 + clip_range),
                ).mean()

                ent_loss = -(entropy if entropy is not None else -log_prob).mean()

                v_loss = self.value_loss_fn(values, rollout_data.returns)

                loss = pi_loss + self.ent_coef * ent_loss + self.vf_coef * v_loss

                self.policy.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                pi_losses.append(pi_loss.item())
                clip_fractions.append(
                    ((ratio - 1).abs() > clip_range).float().mean().item()
                )
                ent_losses.append(ent_loss.item())
                v_losses.append(v_loss.item())
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = (
                        ((log_ratio.exp() - 1) - log_ratio).mean().cpu().numpy()
                    )
                    approx_kl_divs.append(approx_kl_div)

                _params.append(F.softplus(values[..., -1]).detach())

            self._n_updates += 1
            if not continue_training:
                break

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/entropy_loss", np.mean(ent_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pi_losses))
        self.logger.record("train/value_loss", np.mean(v_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/clip_range", clip_range)

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

    def value_loss_fn(self, values: th.Tensor, returns: th.Tensor) -> th.Tensor:
        return mse_loss(values, returns)


class GD_PPO(PPO):
    distribution = "gaussian"

    def value_loss_fn(self, values: th.Tensor, returns: th.Tensor) -> th.Tensor:
        return gaussian_nll_loss(values, returns)


class IV_GD_PPO(GD_PPO):
    def value_loss_fn(self, values: th.Tensor, returns: th.Tensor) -> th.Tensor:
        return gaussian_nll_loss(
            values, returns
        ) + self.uncertainty_temperature * biv_loss(
            values, returns, self.gamma, self.min_batch_size, "mse"
        )


class GGD_PPO(PPO):
    distribution = "ggd"

    def value_loss_fn(self, values: th.Tensor, returns: th.Tensor) -> th.Tensor:
        return ggd_nll_loss(values, returns)


class IV_GGD_PPO(GGD_PPO):
    def value_loss_fn(self, values: th.Tensor, returns: th.Tensor) -> th.Tensor:
        return ggd_nll_loss(values, returns) + self.uncertainty_temperature * biv_loss(
            values, returns, self.gamma, self.min_batch_size, "mae"
        )


class IEV_GGD_PPO(GGD_PPO):
    def value_loss_fn(self, values: th.Tensor, returns: th.Tensor) -> th.Tensor:
        return ggd_nll_loss(values, returns) + self.uncertainty_temperature * biev_loss(
            values, returns, self.min_batch_size, "mae"
        )
