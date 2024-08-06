from .dqn import ALGOS as DQNS
from .ppo import ALGOS as PPOS
from .sac import ALGOS as SACS

ALGOS = DQNS | PPOS | SACS
