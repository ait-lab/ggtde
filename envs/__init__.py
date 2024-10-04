from gymnasium.envs.registration import register

ENVS = {
    # Continuous Control
    "ant": "Ant-v4",
    "half-cheetah": "HalfCheetah-v4",
    "hopper": "Hopper-v4",
    "humanoid": "Humanoid-v4",
    "walker": "Walker2d-v4",
    # Discrete Control
    "cart-pole": "CartPoleNoisy",
    "lunar-lander": "LunarLanderNoisy",
    "mountain-car": "MountainCarNoisy",
}

register(
    id="CartPoleNoisy",
    entry_point="envs.cartpole:CartPoleNoisyEnv",
    vector_entry_point="envs.cartpole:CartPoleNoisyVectorEnv",
    max_episode_steps=500,
    reward_threshold=475,
)

register(
    id="LunarLanderNoisy",
    entry_point="gymnasium.envs.box2d.lunar_lander:LunarLander",
    max_episode_steps=1_000,
    reward_threshold=200,
    kwargs={"continuous": False, "enable_wind": True},
)

register(
    id="MountainCarNoisy",
    entry_point="envs.mountain_car:MountainCarNoisyEnv",
    max_episode_steps=200,
    reward_threshold=-110,
)
