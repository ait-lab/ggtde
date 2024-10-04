from functools import partial
from typing import Any, ClassVar, Dict, List, Literal, Optional, Type, Union

import torch as th
from gymnasium import spaces
from stable_baselines3.common.policies import (
    ContinuousCritic as OriginalContinuousCritic,
)
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.sac.policies import SACPolicy
from torch import nn, optim

from ..common.policies import BasePolicy


class ContinuousCritic(OriginalContinuousCritic):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        ensemble_size: int,
        distribution: Optional[Literal["gaussian", "ggd"]],
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        **kwargs,
    ):
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

        action_dim = get_action_dim(self.action_space)

        self.q_networks: List[nn.ModuleList] = []
        for idx in range(self.n_critics):
            q_net = nn.ModuleList(
                [
                    nn.Sequential(
                        *create_mlp(
                            features_dim + action_dim,
                            self.value_dims,
                            net_arch,
                            activation_fn,
                        )
                    )
                    for _ in range(ensemble_size)
                ]
            )
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, observations: th.Tensor, actions: th.Tensor) -> th.Tensor:
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(observations, self.features_extractor)
        return th.stack(
            [
                th.stack(
                    [net(th.cat([features, actions], dim=1)) for net in q_net],
                    dim=1,
                )
                for q_net in self.q_networks
            ],
        )

    def q1_forward(self, observations: th.Tensor, actions: th.Tensor) -> th.Tensor:
        with th.no_grad():
            features = self.extract_features(observations, self.features_extractor)
        return th.stack(
            [net(th.cat([features, actions], dim=1)) for net in self.q_networks[0]],
            dim=1,
        )


class MlpPolicy(BasePolicy, SACPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        ensemble_size: int,
        distribution: Optional[Literal["gaussian", "ggd"]],
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        log_std_init: float = 0,
        use_expln: bool = False,
        clip_mean: float = 2,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        optimizer_class: Type[optim.Optimizer] = optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
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
            log_std_init=log_std_init,
            use_expln=use_expln,
            clip_mean=clip_mean,
            features_extractor_class=features_extractor_class,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            share_features_extractor=share_features_extractor,
        )

    def make_critic(
        self, features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(
            self.critic_kwargs, features_extractor
        )
        return ContinuousCritic(
            **critic_kwargs,
            ensemble_size=self.ensemble_size,
            distribution=self.distribution,  # type: ignore
        ).to(self.device)


class CnnPolicy(MlpPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        ensemble_size: int,
        distribution: Optional[Literal["gaussian", "ggd"]],
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        log_std_init: float = 0,
        full_std: bool = True,
        use_expln: bool = False,
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
            ortho_init=ortho_init,
            log_std_init=log_std_init,
            full_std=full_std,
            use_expln=use_expln,
            features_extractor_class=NatureCNN,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )


class MultiInputPolicy(MlpPolicy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        ensemble_size: int,
        distribution: Optional[Literal["gaussian", "ggd"]],
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        log_std_init: float = 0,
        full_std: bool = True,
        use_expln: bool = False,
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
            ortho_init=ortho_init,
            log_std_init=log_std_init,
            full_std=full_std,
            use_expln=use_expln,
            features_extractor_class=CombinedExtractor,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )


def get_policy_aliases(
    ensemble_size: int, distribution: Optional[Literal["gaussian", "ggd"]]
) -> ClassVar[Dict[str, Type[SACPolicy]]]:  # type: ignore
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
