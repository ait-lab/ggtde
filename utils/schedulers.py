from typing import Callable

import numpy as np


def exponential_schedule(
    initial_value: float, gamma: float
) -> Callable[[float], float]:
    """Exponential learning rate schedule."""

    return lambda progress_remaining: initial_value * gamma ** (1 - progress_remaining)


def cosine_annealing_schedule(
    initial_value: float, final_value: float
) -> Callable[[float], float]:
    """Cosine annealing learning rate schedule."""

    return lambda progress_remaining: final_value + 0.5 * (
        initial_value - final_value
    ) * (1 + np.cos((1 - progress_remaining) * np.pi))
