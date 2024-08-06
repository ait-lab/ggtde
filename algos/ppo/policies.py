from functools import partial
from typing import Any, ClassVar, Dict, List, Literal, Optional, Type, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
)
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
)
from stable_baselines3.common.type_aliases import Schedule
from torch import nn, optim

from ..common.policies import BasePolicy


class ValueNetwork(nn.Module):
    def __init__(
        self,
        features_dim: int,
        ensemble_size: int,
        distribution: Optional[Literal["gaussian", "ggd"]],
    ):
        self.distribution = distribution
        self.value_dims = 2 - int(self.distribution is None)

        super().__init__()

        self.v_net = nn.ModuleList(
            [nn.Linear(features_dim, self.value_dims) for _ in range(ensemble_size)]
        )

    def forward(self, features: th.Tensor) -> th.Tensor:
        return th.stack([net(features) for net in self.v_net], dim=1)


class MlpPolicy(BasePolicy, ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        ensemble_size: int,
        distribution: Optional[Literal["gaussian", "ggd"]],
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        log_std_init: float = 0,
        full_std: bool = True,
        use_expln: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
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
            ortho_init=ortho_init,
            log_std_init=log_std_init,
            full_std=full_std,
            use_expln=use_expln,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi,
                latent_sde_dim=latent_dim_pi,
                log_std_init=self.log_std_init,
            )
        elif isinstance(
            self.action_dist,
            (
                CategoricalDistribution,
                MultiCategoricalDistribution,
                BernoulliDistribution,
            ),
        ):
            self.action_net = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi
            )
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        self.value_net = ValueNetwork(
            self.mlp_extractor.latent_dim_vf, self.ensemble_size, self.distribution  # type: ignore
        )

        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs  # type: ignore
        )


class CnnPolicy(MlpPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
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
        action_space: spaces.Space,
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
) -> ClassVar[Dict[str, Type[ActorCriticPolicy]]]:  # type: ignore
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
