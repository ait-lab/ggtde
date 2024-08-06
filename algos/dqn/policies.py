from functools import partial
from typing import Any, ClassVar, Dict, List, Literal, Optional, Type

import torch as th
from gymnasium import spaces
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.dqn.policies import QNetwork as OriginalQNetwork
from torch import nn, optim
from torch.nn import functional as F

from ..common.policies import BasePolicy


class QNetwork(OriginalQNetwork):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        ensemble_size: int,
        distribution: Optional[Literal["gaussian", "ggd"]],
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        **kwargs,
    ):
        self.ensemble_size = ensemble_size
        self.distribution = distribution
        self.value_dims = 2 - int(self.distribution is None)

        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            features_dim=features_dim,
            net_arch=net_arch,
            activation_fn=activation_fn,
        )

        self.q_net = nn.ModuleList(
            [
                nn.Sequential(
                    *create_mlp(
                        self.features_dim,
                        int(self.action_space.n) * self.value_dims,
                        self.net_arch,
                        self.activation_fn,
                    )
                )
                for _ in range(ensemble_size)
            ]
        )

    def forward(self, obs: PyTorchObs) -> th.Tensor:
        return th.stack(
            [
                net(self.extract_features(obs, self.features_extractor))
                for net in self.q_net
            ],
            dim=1,
        ).view(-1, self.ensemble_size, int(self.action_space.n), self.value_dims)

    def _predict(
        self, observation: PyTorchObs, deterministic: bool = True
    ) -> th.Tensor:
        values = self(observation)

        if self.distribution == "ggd":
            betas = F.softplus(values[..., 1])
            ensembled_values = (values[..., 0] * betas).mean(dim=1) / betas.sum(dim=1)
        else:
            ensembled_values = values[..., 0].mean(dim=1)

        return ensembled_values.argmax(dim=1)


class MlpPolicy(BasePolicy, DQNPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        ensemble_size: int,
        distribution: Optional[Literal["gaussian", "ggd"]],
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_class: Type[optim.Optimizer] = optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.ensemble_size = ensemble_size
        self.distribution = distribution

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

    def make_q_net(self) -> QNetwork:
        net_args = self._update_features_extractor(
            self.net_args, features_extractor=None
        )
        return QNetwork(
            **net_args, ensemble_size=self.ensemble_size, distribution=self.distribution  # type: ignore
        ).to(self.device)


class CnnPolicy(MlpPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        ensemble_size: int,
        distribution: Optional[Literal["gaussian", "ggd"]],
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_class: Type[optim.Optimizer] = optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            ensemble_size,
            distribution,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )


class MultiInputPolicy(MlpPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        ensemble_size: int,
        distribution: Optional[Literal["gaussian", "ggd"]],
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_class: Type[optim.Optimizer] = optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            ensemble_size,
            distribution,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )


def get_policy_aliases(
    ensemble_size: int, distribution: Optional[Literal["gaussian", "ggd"]]
) -> ClassVar[Dict[str, Type[DQNPolicy]]]:  # type: ignore
    return {
        "MlpPolicy": partial(
            MlpPolicy, ensemble_size=ensemble_size, distribution=distribution
        ),
        "CnnPolicy": partial(
            CnnPolicy, ensemble_size=ensemble_size, distribution=distribution
        ),
        "MultiInputPolicy": partial(
            MultiInputPolicy, ensemble_size=ensemble_size, distribution=distribution
        ),
    }
