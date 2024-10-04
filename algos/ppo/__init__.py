from .algo import GD_PPO, GGD_PPO, IEV_GGD_PPO, IV_GD_PPO, IV_GGD_PPO, PPO

ALGOS = {
    "ppo": PPO,
    "gd-ppo": GD_PPO,
    "iv-gd-ppo": IV_GD_PPO,
    "ggd-ppo": GGD_PPO,
    "iv-ggd-ppo": IV_GGD_PPO,
    "iev-ggd-ppo": IEV_GGD_PPO,
}
