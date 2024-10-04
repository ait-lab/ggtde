import torch as th
from torch.nn import functional as F


def mse_loss(values: th.Tensor, targets: th.Tensor) -> th.Tensor:
    """Calculate mean squared error loss.

    Args:
        values (th.Tensor): Value approximates with shape [batch_size, ensemble_size, 1].
        targets (th.Tensor): TD targets with shape [batch_size, ensemble_size].
    """

    loss = sum(
        F.mse_loss(values[:, i, 0], targets[:, i]) for i in range(values.shape[1])
    )
    assert isinstance(loss, th.Tensor)
    return loss.mean()


def gaussian_nll_loss(values: th.Tensor, targets: th.Tensor) -> th.Tensor:
    """Calculate Gaussian negative log likelihood loss.

    Args:
        values (th.Tensor): Value approximates with shape [batch_size, ensemble_size, 2].
        targets (th.Tensor): TD targets with shape [batch_size, ensemble_size].
    """

    variance = F.softplus(values[..., -1])

    loss = sum(
        F.gaussian_nll_loss(values[:, i, 0], targets[:, i], variance[:, i])
        for i in range(values.shape[1])
    )
    assert isinstance(loss, th.Tensor)
    return loss.mean()


def ggd_nll_loss(values: th.Tensor, targets: th.Tensor) -> th.Tensor:
    """Calculate GGD negative log likelihood loss with risk-averse weighting.

    Args:
        values (th.Tensor): Value approximates with shape [batch_size, ensemble_size, 2].
        targets (th.Tensor): TD targets with shape [batch_size, ensemble_size].
    """

    beta = F.softplus(values[..., -1])

    loss = sum(
        (
            (values[:, i, 0] - targets[:, i]).abs() * beta[:, i]
            - beta[:, i].log()
            + beta[:, i].pow(-1).lgamma()
        )
        * beta[:, i]
        / beta.sum(dim=-1)
        for i in range(values.shape[1])
    )
    assert isinstance(loss, th.Tensor)
    return loss.mean()
