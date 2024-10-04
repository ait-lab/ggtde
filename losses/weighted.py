from typing import Literal

import numpy as np
import torch as th
from scipy.optimize import minimize
from scipy.stats import kurtosis
from torch.nn import functional as F

from utils.constants import EPS


def weighted_loss(
    values: th.Tensor,
    targets: th.Tensor,
    weights: th.Tensor,
    loss_type: Literal["mse", "mae", "huber"],
) -> th.Tensor:
    """Calculate weighted mean squared error loss.

    Args:
        values (th.Tensor): Value approximates with shape [batch_size, ensemble_size, value_dims].
        targets (th.Tensor): TD targets with shape [batch_size, ensemble_size].
        weights (th.Tensor): Weights for the loss with shape [batch_size].
        loss_type (Literal["mse", "mae", "huber"]): Loss type to use.
    """

    if loss_type == "mse":
        loss_func = F.mse_loss
    elif loss_type == "mae":
        loss_func = F.l1_loss
    elif loss_type == "huber":
        loss_func = F.huber_loss
    else:
        raise NotImplementedError(f"Loss type {loss_type} is not implemented.")

    loss = sum(
        loss_func(values[:, i, 0], targets[:, i], reduction="none") * weights
        for i in range(values.shape[1])
    )
    assert isinstance(loss, th.Tensor)
    return loss.mean()


def optimal_xi(variance: np.ndarray, batch_size: int, min_batch_size: int) -> float:
    def effective_batch_size(variance: np.ndarray) -> float:
        inverse_variance = np.power(variance, -1)
        weights = inverse_variance / np.sum(inverse_variance)
        return 1 / np.sum(np.power(weights, 2))

    xi = 0
    if effective_batch_size(variance) < (
        min_batch_size := min(batch_size - 1, min_batch_size)
    ):
        epsilon = minimize(
            lambda xi: abs(effective_batch_size(variance + abs(xi)) - min_batch_size),
            0,
            method="Nelder-Mead",
            options={"fatol": 1, "maxiter": 100},
        )
        if (maybe_xi := epsilon.x) is not None:
            xi = abs(maybe_xi[0])
    return xi


def biv_loss(
    values: th.Tensor,
    targets: th.Tensor,
    gamma: float,
    min_batch_size: int,
    loss_type: Literal["mse", "mae", "huber"],
) -> th.Tensor:
    """Calculate batch inverse variance loss.

    Args:
        values (th.Tensor): Value approximates with shape [batch_size, ensemble_size, value_dims].
        targets (th.Tensor): TD targets with shape [batch_size, ensemble_size].
        gamma (float): Discount factor.
        min_batch_size (int): Minimum effective batch size for batch inverse variance weighting.
        loss_type (Literal["mse", "mae", "huber"]): Loss type to use.

    Note:
        The [original implementation](https://github.com/montrealrobotics/iv_rl/blob/0f72a8f077a238237027ea96b7d1160c35ac9959/dqn/ensembleDQN.py#L121)
        calculates target variance using `th.var(unbiased=True)`. Instead, we use `np.var(ddof=1)` for Bessel's correction.
    """

    batch_size, *_ = values.shape

    approx = values[..., 0].detach().cpu().numpy()
    variance = np.var(approx, axis=-1, ddof=1).clip(EPS) * gamma**2

    inverse_variance = np.power(
        variance + optimal_xi(variance, batch_size, min_batch_size), -1
    )
    biv_weights = th.from_numpy(inverse_variance / np.sum(inverse_variance)).to(
        values.device
    )

    return weighted_loss(values, targets, biv_weights, loss_type)


def biev_loss(
    values: th.Tensor,
    targets: th.Tensor,
    min_batch_size: int,
    loss_type: Literal["mse", "mae", "huber"],
) -> th.Tensor:
    """Calculate batch inverse error variance loss.

    Args:
        values (th.Tensor): Value approximates with shape [batch_size, ensemble_size, value_dims].
        targets (th.Tensor): TD targets with shape [batch_size, ensemble_size].
        min_batch_size (int): Minimum effective batch size for batch inverse variance weighting.
        loss_type (Literal["mse", "mae", "huber"]): Loss type to use.
    """

    batch_size, ensemble_size, _ = values.shape

    td_errors = (values[..., 0] - targets).detach().cpu().numpy()
    variance = (
        np.var(td_errors, axis=-1, ddof=0)
        / (
            kurtosis(td_errors, axis=-1, bias=False) / ensemble_size
            + (ensemble_size + 1) / (ensemble_size - 1)
        )
    ).clip(EPS)

    inverse_variance = np.power(
        variance + optimal_xi(variance, batch_size, min_batch_size), -1
    )
    biev_weights = th.from_numpy(inverse_variance / np.sum(inverse_variance)).to(
        values.device
    )

    return weighted_loss(values, targets, biev_weights, loss_type)
