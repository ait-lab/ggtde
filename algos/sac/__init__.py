from .algo import GD_SAC, GGD_SAC, IEV_GGD_SAC, IV_GD_SAC, IV_GGD_SAC, SAC

ALGOS = {
    "sac": SAC,
    "gd-sac": GD_SAC,
    "iv-gd-sac": IV_GD_SAC,
    "ggd-sac": GGD_SAC,
    "iv-ggd-sac": IV_GGD_SAC,
    "iev-ggd-sac": IEV_GGD_SAC,
}
