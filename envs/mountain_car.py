import math

import numpy as np
from gymnasium.envs.classic_control.mountain_car import MountainCarEnv

from .base import NoisyEnv


class MountainCarNoisyEnv(NoisyEnv, MountainCarEnv):
    def __init__(self, max_noise: float = 0.3):
        super().__init__()

        self.max_noise = max_noise

    def step(self, action: int):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"

        force = (action - 1) * self.noisy(self.force)

        position, velocity = self.state
        velocity += force + math.cos(3 * position) * (-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if position == self.min_position and velocity < 0:
            velocity = 0

        terminated = bool(
            position >= self.goal_position and velocity >= self.goal_velocity
        )
        reward = -1

        self.state = (position, velocity)
        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}
