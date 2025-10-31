"""World model training utilities for TokenHSI."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

from tokenhsi.utils import traj_generator


@dataclasses.dataclass
class WorldModelConfig:
    """Configuration for the trajectory world model trainer."""

    cfg_env_path: str
    output_path: str
    prediction_horizon: int = 1
    num_envs: int = 512
    num_batches: int = 32
    batch_size: int = 1024
    epochs: int = 10
    learning_rate: float = 1e-3
    hidden_dim: int = 256
    device: str = "auto"
    seed: int = 13


class TrajectoryWorldModel(nn.Module):
    """A small MLP that predicts the next trajectory vertex."""

    def __init__(self, input_dim: int = 3, hidden_dim: int = 256, output_dim: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _load_env_config(cfg_env_path: str) -> Dict:
    with open(cfg_env_path, "r", encoding="utf-8") as fh:
        return yaml.load(fh, Loader=yaml.SafeLoader)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class WorldModelTrainer:
    """Trainer that learns a 1-step predictive model over generated trajectories."""

    def __init__(self, config: WorldModelConfig) -> None:
        self.cfg = config
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        self.device = _resolve_device(self.cfg.device)

        env_cfg = _load_env_config(self.cfg.cfg_env_path)
        self.env_params = env_cfg.get("env", {})

        episode_length = self.env_params.get("episodeLength", 300)
        control_frequency_inv = self.env_params.get("controlFrequencyInv", 2)
        # Isaac Gym default dt is 1/60.0. ControlFrequencyInv specifies the action update interval.
        sim_dt = 1.0 / 60.0 * control_frequency_inv
        episode_dur = episode_length * sim_dt

        num_verts = 101
        dtheta_max = 2.0
        self._traj_gen = traj_generator.TrajGenerator(
            self.cfg.num_envs,
            episode_dur,
            num_verts,
            self.device,
            dtheta_max,
            self.env_params.get("speedMin", 0.5),
            self.env_params.get("speedMax", 1.5),
            self.env_params.get("accelMax", 2.0),
            self.env_params.get("sharpTurnProb", 0.02),
            self.env_params.get("sharpTurnAngle", math.pi / 2.0),
        )

    def _generate_dataset(self) -> TensorDataset:
        if self.cfg.prediction_horizon < 1:
            raise ValueError("prediction_horizon must be >= 1")

        env_ids = torch.arange(self.cfg.num_envs, dtype=torch.long, device=self.device)
        horizon = self.cfg.prediction_horizon

        features: List[torch.Tensor] = []
        targets: List[torch.Tensor] = []

        for _ in range(self.cfg.num_batches):
            init_pos = torch.zeros((self.cfg.num_envs, 3), device=self.device)
            init_pos[:, :2] = torch.empty((self.cfg.num_envs, 2), device=self.device).uniform_(-2.0, 2.0)
            self._traj_gen.reset(env_ids, init_pos)

            verts = self._traj_gen.get_traj_verts(env_ids)
            verts = verts[..., :3]
            current = verts[:, :-horizon, :]
            future = verts[:, horizon:, :]

            features.append(current.reshape(-1, current.shape[-1]).cpu())
            targets.append(future.reshape(-1, future.shape[-1]).cpu())

        inputs = torch.cat(features, dim=0)
        outputs = torch.cat(targets, dim=0)
        return TensorDataset(inputs, outputs)

    def train(self) -> Dict[str, float]:
        os.makedirs(os.path.dirname(self.cfg.output_path), exist_ok=True)

        dataset = self._generate_dataset()
        loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True)

        model = TrajectoryWorldModel(hidden_dim=self.cfg.hidden_dim)
        model.to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.learning_rate)

        model.train()
        final_loss = 0.0
        for epoch in range(self.cfg.epochs):
            epoch_loss = 0.0
            for batch_inputs, batch_targets in loader:
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)

                preds = model(batch_inputs)
                loss = criterion(preds, batch_targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_inputs.size(0)

            epoch_loss /= len(loader.dataset)
            final_loss = epoch_loss
            print(f"[WorldModel] Epoch {epoch + 1}/{self.cfg.epochs} - loss: {epoch_loss:.6f}")

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": dataclasses.asdict(self.cfg),
                "env_params": self.env_params,
            },
            self.cfg.output_path,
        )

        return {"loss": final_loss}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight trajectory world model.")
    parser.add_argument("--cfg_env", required=True, help="Environment configuration used for trajectory parameters.")
    parser.add_argument("--output_path", required=True, help="Path to save the trained world model checkpoint.")
    parser.add_argument("--prediction_horizon", type=int, default=1, help="Number of frames ahead to predict.")
    parser.add_argument("--num_envs", type=int, default=512, help="Number of parallel trajectories to sample per batch.")
    parser.add_argument("--num_batches", type=int, default=32, help="Number of trajectory batches to sample.")
    parser.add_argument("--batch_size", type=int, default=1024, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for the optimizer.")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension of the MLP world model.")
    parser.add_argument("--device", type=str, default="auto", help="Computation device (cpu, cuda, or auto).")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for reproducibility.")

    args = parser.parse_args()

    config = WorldModelConfig(
        cfg_env_path=args.cfg_env,
        output_path=args.output_path,
        prediction_horizon=args.prediction_horizon,
        num_envs=args.num_envs,
        num_batches=args.num_batches,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        device=args.device,
        seed=args.seed,
    )

    trainer = WorldModelTrainer(config)
    metrics = trainer.train()
    print(json.dumps({"loss": metrics["loss"], "output": config.output_path}))


if __name__ == "__main__":
    main()
