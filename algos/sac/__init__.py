from .algo import (
    GD_SAC,
    GGD_SAC,
    IEV_GD_SAC,
    IEV_GGD_SAC,
    IEV_SAC,
    IV_GD_SAC,
    IV_GGD_SAC,
    IV_SAC,
    SAC,
)

ALGOS = {
    "sac": SAC,
    "iv-sac": IV_SAC,
    "iev-sac": IEV_SAC,
    "gd-sac": GD_SAC,
    "iv-gd-sac": IV_GD_SAC,
    "iev-gd-sac": IEV_GD_SAC,
    "ggd-sac": GGD_SAC,
    "iv-ggd-sac": IV_GGD_SAC,
    "iev-ggd-sac": IEV_GGD_SAC,
}
