"""World model utilities for predicting the next observation frame."""

from __future__ import annotations

from typing import Iterable, List

import torch
from torch import nn


def _build_mlp(input_dim: int, hidden_dims: Iterable[int], activation: str) -> nn.Sequential:
    """Constructs an MLP that maps ``input_dim`` to itself."""
    layers: List[nn.Module] = []
    last_dim = input_dim
    act_layer = _activation_from_name(activation)

    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(act_layer())
        last_dim = hidden_dim

    layers.append(nn.Linear(last_dim, input_dim))
    return nn.Sequential(*layers)


def _activation_from_name(name: str):
    """Returns an activation constructor from a lowercase name."""
    if name is None:
        return nn.Identity

    name = name.lower()
    if name == "relu":
        return nn.ReLU
    if name == "elu":
        return nn.ELU
    if name == "tanh":
        return nn.Tanh
    if name == "gelu":
        return nn.GELU
    if name in ("identity", "linear", "none"):
        return nn.Identity

    raise ValueError(f"Unsupported activation '{name}'.")


class WorldModel(nn.Module):
    """Predicts the next observation given the current observation."""

    def __init__(self, obs_dim: int, hidden_dims: Iterable[int], activation: str = "relu"):
        super().__init__()
        self.obs_dim = obs_dim
        self.net = _build_mlp(obs_dim, hidden_dims, activation)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(obs)
