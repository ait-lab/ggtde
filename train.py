from argparse import ArgumentParser, Namespace
from datetime import datetime

import torch
import wandb

from algos import ALGOS
from envs import ENVS
from utils.trainer import Trainer


def train(args: Namespace):
    wandb.init(
        config=vars(args),
        name=datetime.now().strftime("[%m-%d]%H.%M.%S"),
        sync_tensorboard=True,
        monitor_gym=True,
    )

    trainer = Trainer(args)

    if (results := trainer.setup_experiment()) is not None:
        model, saved_hyperparams = results
        args.saved_hyperparams = saved_hyperparams

        if model is not None:
            trainer.learn(model)
            trainer.save_trained_model(model)
    else:
        trainer.hyperparameters_optimization()


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--algo",
        type=str,
        choices=ALGOS.keys(),
        required=True,
        help="The algorithm to use",
    )
    parser.add_argument(
        "--env",
        type=str,
        choices=ENVS.keys(),
        required=True,
        help="The environment to train on",
    )
    parser.add_argument(
        "--trained-path",
        type=str,
        default="",
        help="The path to the trained agent",
        dest="trained_agent",
    )
    parser.add_argument("--seed", type=int, default=718, help="The random seed")
    parser.add_argument(
        "--verbose", type=int, choices=[0, 1], default=0, help="The verbosity level"
    )
    parser.add_argument("--device", type=str, default="auto", help="The device to use")
    args = parser.parse_args()

    if not (
        hasattr(torch.backends, args.device)
        and getattr(torch.backends, args.device).is_available()
    ):
        args.device = "auto"

    train(args)


if __name__ == "__main__":
    main()
