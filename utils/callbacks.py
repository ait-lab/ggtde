import os
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import EvalCallback as OriginalEvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv, sync_envs_normalization


class EvalCallback(OriginalEvalCallback):
    def __init__(
        self,
        eval_env: VecEnv,
        n_eval_episodes: int = 5,
        eval_freq: int = 10_000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        name_prefix: str = "rl_model",
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
    ):
        super().__init__(
            eval_env=eval_env,
            callback_on_new_best=None,
            callback_after_eval=None,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            deterministic=deterministic,
            render=render,
            verbose=verbose,
            warn=verbose > 0,
        )

        self.name_prefix = name_prefix

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way"
                    ) from e

            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=self.deterministic,
                render=self.render,
                callback=self._log_success_callback,
                return_episode_rewards=True,
                warn=self.warn,
            )

            if self.log_path is not None:
                assert isinstance(episode_rewards, list)
                assert isinstance(episode_lengths, list)
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = {"successes": self.evaluations_successes}

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            self.logger.record(
                "eval/avg_reward", avg_reward := np.mean(episode_rewards)
            )
            self.logger.record("eval/mdn_reward", np.median(episode_rewards))
            self.logger.record(
                "eval/avg_epilen", avg_length := np.mean(episode_lengths)
            )
            self.logger.record("eval/mdn_epilen", np.median(episode_lengths))
            self.last_mean_reward = avg_reward

            if self.verbose >= 1:
                std_reward, std_length = np.std(episode_rewards), np.std(
                    episode_lengths
                )
                print(
                    f"At {self.num_timesteps:,d} step:\n"
                    f"Episode reward: {avg_reward:.2f} +/- {std_reward:.2f} | "
                    f"Episode length: {avg_length:,.2f} +/- {std_length:,.2f}"
                )

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {success_rate:.2%}")
                self.logger.record("eval/success_rate", success_rate)

            self.logger.record(
                "time/training_steps", self.num_timesteps, exclude="tensorboard"
            )
            self.logger.dump(self.num_timesteps)

            if avg_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(
                        os.path.join(
                            self.best_model_save_path,
                            f"{self.name_prefix}_{self.num_timesteps}_best",
                        )
                    )
                self.best_mean_reward = float(avg_reward)
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training
