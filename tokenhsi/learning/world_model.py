"""World model utilities for AMP-based single-task policies.

This module provides a light-weight dynamics model that predicts the next
environment state and reward given the current state and the action proposed by
an AMP policy.  The model can be trained on rollouts collected from the real
Isaac Gym environment and later used to perform short-horizon imagination or to
warm start policies in data-sparse regimes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class WorldModelConfig:
    """Configuration for :class:`AMPWorldModel`.

    Attributes:
        state_dim: Number of state features provided to the policy.
        action_dim: Number of action features produced by the policy.
        hidden_dims: Width of the hidden layers for the dynamics MLP.
        activation: Callable used after every hidden linear layer.
        predict_delta: If ``True`` the model predicts a delta that is added to
            the current state. Otherwise, the model predicts the absolute next
            state directly.
    """

    state_dim: int
    action_dim: int
    hidden_dims: Sequence[int] = (512, 512)
    activation: Callable[[torch.Tensor], torch.Tensor] = F.elu
    predict_delta: bool = True


class AMPWorldModel(nn.Module):
    """A small multi-layer perceptron that learns environment dynamics.

    The model receives the policy action, the current state, and (optionally)
    the most recent reward to produce a prediction of the next state and reward.
    The interface is intentionally minimal so that it can be plugged into the
    existing AMP training pipelines without changes to the rl-games core.  The
    model can be trained with standard supervised objectives using data sampled
    from the replay buffer or on-policy rollouts.
    """

    def __init__(self, config: WorldModelConfig):
        super().__init__()

        self._config = config
        input_dim = config.state_dim + config.action_dim + 1  # +1 for reward
        hidden_dims: Iterable[int] = list(config.hidden_dims)

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        self._hidden_layers = nn.ModuleList(layers)

        self._state_head = nn.Linear(prev_dim, config.state_dim)
        self._reward_head = nn.Linear(prev_dim, 1)

        self.reset_parameters()

    @property
    def config(self) -> WorldModelConfig:
        return self._config

    def reset_parameters(self) -> None:
        """Applies a small orthogonal initialisation to stabilise training."""

        gain = nn.init.calculate_gain("relu")
        for layer in self._hidden_layers:
            nn.init.orthogonal_(layer.weight, gain)
            nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self._state_head.weight, gain)
        nn.init.zeros_(self._state_head.bias)
        nn.init.orthogonal_(self._reward_head.weight, gain)
        nn.init.zeros_(self._reward_head.bias)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict the next state and reward.

        Args:
            state: Current state tensor with shape ``[B, state_dim]``.
            action: Action tensor with shape ``[B, action_dim]``.
            reward: Optional reward tensor with shape ``[B]`` or
                ``[B, 1]`` representing the most recent reward signal.

        Returns:
            A tuple ``(next_state, next_reward)``. ``next_state`` matches the
            dimensionality of ``state``; ``next_reward`` has shape ``[B]``.
        """

        if reward is None:
            reward_input = torch.zeros_like(state[:, :1])
        else:
            reward_input = reward.unsqueeze(-1) if reward.dim() == 1 else reward

        x = torch.cat([state, action, reward_input], dim=-1)

        for layer in self._hidden_layers:
            x = self._config.activation(layer(x))

        state_delta = self._state_head(x)
        reward_pred = self._reward_head(x).squeeze(-1)

        if self._config.predict_delta:
            next_state = state + state_delta
        else:
            next_state = state_delta

        return next_state, reward_pred

    @torch.no_grad()
    def imagine_step(
        self,
        policy: Callable[[torch.Tensor], torch.Tensor],
        state: torch.Tensor,
        reward: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Roll out the world model for a single step.

        Args:
            policy: Callable that maps states to actions. The callable is
                typically the policy network used for AMP training.
            state: Current state batch.
            reward: Optional last reward batch.

        Returns:
            A tuple ``(action, next_state, predicted_reward)`` describing the
            world-model rollout.
        """

        action = policy(state)
        next_state, predicted_reward = self(state, action, reward)
        return action, next_state, predicted_reward

    def loss(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        reduction: str = "mean",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute supervised losses for a batch of transitions.

        Args:
            batch: Tuple ``(state, action, reward, next_state)`` collected from
                the environment.
            reduction: Either ``"mean"`` or ``"sum"``.

        Returns:
            Tuple containing the state loss, reward loss, and the total loss.
        """

        state, action, reward, next_state = batch
        predicted_next_state, predicted_reward = self(state, action, reward)

        state_loss = F.mse_loss(predicted_next_state, next_state, reduction=reduction)
        reward_loss = F.mse_loss(predicted_reward, reward, reduction=reduction)
        total_loss = state_loss + reward_loss
        return state_loss, reward_loss, total_loss

    def update(
        self,
        optimizer: torch.optim.Optimizer,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        gradient_clip: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        """Perform a single optimisation step using the provided batch."""

        state_loss, reward_loss, total_loss = self.loss(batch)
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        if gradient_clip is not None:
            nn.utils.clip_grad_norm_(self.parameters(), gradient_clip)
        optimizer.step()
        return (
            state_loss.detach().item(),
            reward_loss.detach().item(),
            total_loss.detach().item(),
        )


def step_env_with_world_model(
    env,
    world_model: AMPWorldModel,
    policy: Callable[[torch.Tensor], torch.Tensor],
    state: torch.Tensor,
    reward: Optional[torch.Tensor] = None,
):
    """Utility to use the world model to propose an action for the environment.

    The helper mirrors the diagram in the user-provided AMP figure: the current
    state is fed into the policy to get an action, the world model predicts the
    next state and reward, and the environment is stepped with the very same
    action.  The function returns both the model prediction and the real
    environment feedback so that callers can measure model accuracy online.
    """

    action, imagined_next_state, imagined_reward = world_model.imagine_step(
        policy, state, reward
    )

    env_next_state, env_reward, done, info = env.step(action)
    return {
        "action": action,
        "imagined_next_state": imagined_next_state,
        "imagined_reward": imagined_reward,
        "env_next_state": env_next_state,
        "env_reward": env_reward,
        "done": done,
        "info": info,
    }
