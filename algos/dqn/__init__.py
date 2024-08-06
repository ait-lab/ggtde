from .algo import (
    DQN,
    GD_DQN,
    GGD_DQN,
    IEV_DQN,
    IEV_GD_DQN,
    IEV_GGD_DQN,
    IV_DQN,
    IV_GD_DQN,
    IV_GGD_DQN,
)

ALGOS = {
    "dqn": DQN,
    "iv-dqn": IV_DQN,
    "iev-dqn": IEV_DQN,
    "gd-dqn": GD_DQN,
    "iv-gd-dqn": IV_GD_DQN,
    "iev-gd-dqn": IEV_GD_DQN,
    "ggd-dqn": GGD_DQN,
    "iv-ggd-dqn": IV_GGD_DQN,
    "iev-ggd-dqn": IEV_GGD_DQN,
}
