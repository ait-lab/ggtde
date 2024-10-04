import math
from typing import Optional, Tuple

import numpy as np
from gymnasium import logger
from gymnasium.envs.classic_control.cartpole import CartPoleEnv, CartPoleVectorEnv

from .base import NoisyEnv


class CartPoleNoisyEnv(NoisyEnv, CartPoleEnv):
    def __init__(self, max_noise: float = 0.3, render_mode: Optional[str] = None):
        super().__init__(render_mode)

        self.max_noise = max_noise

    def step(self, action: int):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."
        x, x_dot, theta, theta_dot = self.state
        costheta, sintheta = math.cos(theta), math.sin(theta)

        force = (action - 0.5) / abs(action - 0.5) * self.noisy(self.force_mag)

        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4 / 3 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not terminated:
            reward = 1
        elif self.steps_beyond_terminated is None:
            self.steps_beyond_terminated = 0
            reward = 1
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            reward = 0

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}


class CartPoleNoisyVectorEnv(NoisyEnv, CartPoleVectorEnv):
    def __init__(
        self,
        max_noise: float = 0.3,
        num_envs: int = 2,
        max_episode_steps: int = 500,
        render_mode: Optional[str] = None,
    ):
        super().__init__(num_envs, max_episode_steps, render_mode)

        self.max_noise = max_noise

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."

        x, x_dot, theta, theta_dot = self.state
        costheta, sintheta = np.cos(theta), np.sin(theta)

        force = np.sign(action - 0.5) * self.noisy(self.force_mag)

        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4 / 3 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = np.stack((x, x_dot, theta, theta_dot))

        terminated: np.ndarray = (
            (x < -self.x_threshold)
            | (x > self.x_threshold)
            | (theta < -self.theta_threshold_radians)
            | (theta > self.theta_threshold_radians)
        )

        self.steps += 1

        truncated = self.steps >= self.max_episode_steps

        done = terminated | truncated

        if any(done):
            self.state[:, done] = self.np_random.uniform(
                low=self.low, high=self.high, size=(4, done.sum())
            ).astype(np.float32)
            self.steps[done] = 0

        reward = np.ones_like(terminated, dtype=np.float32)

        if self.render_mode == "human":
            self.render()

        return self.state.T, reward, terminated, truncated, {}
