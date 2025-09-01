"""Example PPO training script for TokenHSI environments.

This script mirrors the structure of :mod:`tokenhsi.run` but configures the
:class:`rl_games.torch_runner.Runner` to use the standard PPO algorithm.  It
provides a minimal entry point for training a policy on any environment
supported by the repository.

Usage:
    python -m tokenhsi.train_ppo --task <TaskName> --cfg_train <path/to/train_cfg> \
        --cfg_env <path/to/env_cfg> --num_envs <N> [--headless]

The command line arguments are identical to ``tokenhsi.run``.
"""

from __future__ import annotations

import numpy as np
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

from utils.config import (get_args, load_cfg, parse_sim_params,
                          set_np_formatting, set_seed)
from utils.parse_task import parse_task


args = None
cfg = None
cfg_train = None


def create_rlgpu_env(**kwargs):
    """Create the Isaac Gym environment used for training."""
    sim_params = parse_sim_params(args, cfg, cfg_train)
    task, env = parse_task(args, cfg, cfg_train, sim_params)
    return env


class RLGPUEnv(vecenv.IVecEnv):
    """A thin wrapper that exposes the Isaac Gym environment to rl-games."""

    def __init__(self, config_name, num_actors, **kwargs):
        self.env = env_configurations.configurations[config_name]["env_creator"](**kwargs)
        self.use_global_obs = self.env.num_states > 0
        self.full_state = {"obs": self.reset()}
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.full_state["obs"] = obs
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
            return self.full_state, rew, done, info
        return self.full_state["obs"], rew, done, info

    def reset(self, env_ids=None):
        self.full_state["obs"] = self.env.reset(env_ids)
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
            return self.full_state
        return self.full_state["obs"]

    def get_env_info(self):
        info = {
            "action_space": self.env.action_space,
            "observation_space": self.env.observation_space,
        }
        if self.use_global_obs:
            info["state_space"] = self.env.state_space
        return info


vecenv.register(
    "RLGPU", lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs)
)
env_configurations.register(
    "rlgpu",
    {"env_creator": lambda **kwargs: create_rlgpu_env(**kwargs), "vecenv_type": "RLGPU"},
)


def main() -> None:
    """Launch PPO training using rl-games."""
    global args, cfg, cfg_train

    set_np_formatting()
    args = get_args()
    cfg, cfg_train, _ = load_cfg(args)

    # Ensure reproducibility and configure the PPO algorithm
    cfg_train["params"]["seed"] = set_seed(cfg_train["params"].get("seed", -1), False)
    cfg_train["params"]["algo"]["name"] = "ppo"
    cfg_train["params"]["model"]["name"] = "ppo"
    cfg_train["params"]["network"]["name"] = "mlp"

    runner = Runner()
    runner.load(cfg_train)
    runner.reset()
    runner.run(vars(args))


if __name__ == "__main__":
    main()
