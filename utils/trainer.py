import argparse
from os import path as osp
from typing import Any, Dict, Optional, Tuple

from rl_zoo3.exp_manager import ExperimentManager
from rl_zoo3.utils import linear_schedule
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.utils import constant_fn as constant_schedule
from stable_baselines3.common.vec_env import VecEnv
from wandb.integration.sb3 import WandbCallback

from algos import ALGOS
from envs import ENVS

from .callbacks import EvalCallback
from .constants import EPS
from .paths import LOG_DIR, ROOT_DIR
from .schedulers import cosine_annealing_schedule, exponential_schedule


class Trainer(ExperimentManager):
    def __init__(self, args: argparse.Namespace):
        super().__init__(
            args,
            args.algo,
            ENVS[args.env],
            LOG_DIR.as_posix(),
            eval_freq=10_000,
            trained_agent=args.trained_agent,
            seed=args.seed,
            log_interval=1_000,
            verbose=args.verbose,
            vec_env_type="subproc",
            device=args.device,
            config=(
                ROOT_DIR
                / "configs"
                / (args.algo if "-" not in args.algo else args.algo.split("-")[-1])
            )
            .with_suffix(".yml")
            .as_posix(),
            show_progress=True,
        )

        self.tensorboard_log = osp.join(self.log_folder, self.env_name)
        self.log_path = self.save_path = self.params_path = osp.join(
            self.log_folder, self.env_name, self.algo.upper()
        )

    def setup_experiment(self) -> Optional[Tuple[BaseAlgorithm, Dict[str, Any]]]:
        unprocessed_hyperparams, saved_hyperparams = self.read_hyperparameters()
        hyperparams, self.env_wrapper, self.callbacks, self.vec_env_wrapper = (
            self._preprocess_hyperparams(unprocessed_hyperparams)
        )

        self.create_log_folder()
        self.create_callbacks()

        env = self.create_envs(self.n_envs, no_log=self.verbose < 1)

        self._hyperparams = self._preprocess_action_noise(
            hyperparams, saved_hyperparams, env
        )

        if self.continue_training:
            model = self._load_pretrained_agent(self._hyperparams, env)
        else:
            model = ALGOS[self.algo](
                env=env,
                tensorboard_log=self.tensorboard_log,
                verbose=self.verbose,
                seed=self.seed,
                device=self.device,
                **self._hyperparams,
            )

        self._save_config(saved_hyperparams)
        return model, saved_hyperparams

    @staticmethod
    def _preprocess_schedules(hyperparams: Dict[str, Any]) -> Dict[str, Any]:
        for key in ["learning_rate", "clip_range", "clip_range_vf", "delta_std"]:
            if key not in hyperparams:
                continue
            if isinstance(hyperparams[key], (float, int)):
                if (value := float(hyperparams[key])) < 0:
                    continue
                hyperparams[key] = constant_schedule(value)
            elif isinstance(hyperparams[key], str):
                schedule, initial_value, *schedule_args = hyperparams[key].split("_")
                initial_value = float(initial_value)
                if schedule.startswith("lin"):
                    hyperparams[key] = linear_schedule(initial_value)
                elif schedule.startswith("exp"):
                    gamma = float(schedule_args[0]) if schedule_args else EPS
                    hyperparams[key] = exponential_schedule(initial_value, gamma)
                elif schedule.startswith("cos"):
                    final_value = float(schedule_args[0]) if schedule_args else EPS
                    hyperparams[key] = cosine_annealing_schedule(
                        initial_value, final_value
                    )
                else:
                    raise ValueError(f"Invalid value for {key}: {hyperparams[key]}")
        return hyperparams

    def create_callbacks(self):
        self.callbacks.extend(
            [
                ProgressBarCallback(),
                WandbCallback(),
                EvalCallback(
                    self.create_envs(max(self.n_envs // 5, 1), eval_env=True),
                    n_eval_episodes=30,
                    eval_freq=min(self.n_timesteps // 10_000, self.eval_freq),
                    log_path=self.save_path,
                    best_model_save_path=self.save_path,
                    deterministic=self.deterministic_eval,
                    verbose=self.verbose,
                ),
            ]
        )

    def _load_pretrained_agent(
        self, hyperparams: Dict[str, Any], env: VecEnv
    ) -> BaseAlgorithm:
        print("Loading pretrained agent")
        del hyperparams["policy"]

        if "policy_kwargs" in hyperparams.keys():
            del hyperparams["policy_kwargs"]

        model = ALGOS[self.algo].load(
            self.trained_agent,
            env=env,
            tensorboard_log=self.tensorboard_log,
            verbose=self.verbose,
            seed=self.seed,
            device=self.device,
            **hyperparams,
        )

        replay_buffer_path = osp.join(
            osp.dirname(self.trained_agent), "replay_buffer.pkl"
        )

        if osp.exists(replay_buffer_path):
            print("Loading replay buffer")
            assert hasattr(
                model, "load_replay_buffer"
            ), "The current model doesn't have a `load_replay_buffer` to load the replay buffer"
            model.load_replay_buffer(
                replay_buffer_path, truncate_last_traj=self.truncate_last_trajectory
            )
        return model
