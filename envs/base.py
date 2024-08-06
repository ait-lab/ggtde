from gymnasium import Env


class NoisyEnv(Env):
    max_noise: float

    def noisy(self, force: float) -> float:
        return force * (
            1 + self.np_random.uniform(1 - self.max_noise, 1 + self.max_noise)
        )
