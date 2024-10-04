# Generalized Gaussian Temporal Difference Error

Official PyTorch Implementation of [Generalized Gaussian Temporal Difference Error for Uncertainty-aware Reinforcement Learning](https://arxiv.org/abs/2408.02295).

## Usage

### Environment

We manage our packages using `poetry`.
Please refer to the [official guide](https://python-poetry.org/docs/#installation) for installation instructions.
Once installed, run the command `poetry install` to install all the required dependencies.

### Train

> Note: The overall training pipeline follows the structure of [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo).

```shell
poetry run python train.py --algo <RL_ALGORITHM>
                           --env <ENVIRONMENT>
                           --trained-path <TRAINED_MODEL_PATH>
                           --seed <SEED>
                           --verbose <VERBOSE_LEVEL>
```

Arguments:

- `RL_ALGORITHM`: The algorithm to use.
- `ENVIRONMENT`: The environment to train on.
- `TRAINED_MODEL_PATH`: The path to the trained agent.
- `SEED`: The random seed.
- `VERBOSE_LEVEL`: The verbosity level.

## Citation

```bib
@misc{kim2024ggtde,
  title={Generalized Gaussian Temporal Difference Error for Uncertainty-aware Reinforcement Learning},
  author={Seyeon Kim and Joonhun Lee and Namhoon Cho and Sungjun Han and Wooseop Hwang},
  year={2024}
}
```
