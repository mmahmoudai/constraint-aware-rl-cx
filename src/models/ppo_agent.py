"""
PPO-based RL agent for adaptive augmentation policy learning.

Architecture:
    State input: 100-dimensional feature vector
    Shared MLP trunk: 3 × 256 hidden units, ReLU
    Policy head: 256 -> 60 (Softmax)
    Value head: 256 -> 1 (Linear)
    Total parameters: 173,117
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class SharedMLPTrunk(nn.Module):
    """Shared feature extractor: 3 hidden layers of 256 units with ReLU."""

    def __init__(self, state_dim: int = 100, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PolicyHead(nn.Module):
    """Maps shared features to action probabilities over 60 discrete actions."""

    def __init__(self, hidden_dim: int = 256, n_actions: int = 60):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.fc(x), dim=-1)


class ValueHead(nn.Module):
    """Maps shared features to a scalar state-value estimate."""

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)


class PPOAgent(nn.Module):
    """PPO agent with shared trunk, separate policy and value heads.

    Action space: 60 discrete actions = 5 augmentation types × 12 intensity levels
    State space: 100-dimensional modality-aware vector
    Total parameters: 173,117
    """

    def __init__(
        self,
        state_dim: int = 100,
        n_actions: int = 60,
        hidden_dim: int = 256,
        clip_eps: float = 0.2,
        entropy_coeff: float = 0.02,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        actor_lr: float = 1e-4,
        critic_lr: float = 5e-4,
        buffer_size: int = 1024,
        batch_size: int = 128,
    ):
        super().__init__()
        self.trunk = SharedMLPTrunk(state_dim, hidden_dim)
        self.policy_head = PolicyHead(hidden_dim, n_actions)
        self.value_head = ValueHead(hidden_dim)

        self.clip_eps = clip_eps
        self.entropy_coeff = entropy_coeff
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.actor_optimizer = torch.optim.Adam(
            list(self.trunk.parameters()) + list(self.policy_head.parameters()),
            lr=actor_lr,
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.trunk.parameters()) + list(self.value_head.parameters()),
            lr=critic_lr,
        )

        # experience buffer
        self._reset_buffer()

    def _reset_buffer(self):
        self.buffer = {
            "states": [],
            "actions": [],
            "log_probs": [],
            "rewards": [],
            "values": [],
            "dones": [],
        }

    def forward(self, state: torch.Tensor):
        features = self.trunk(state)
        action_probs = self.policy_head(features)
        value = self.value_head(features)
        return action_probs, value

    @torch.no_grad()
    def select_action(self, state: torch.Tensor):
        """Sample an action from the policy and return action, log_prob, value."""
        action_probs, value = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def decode_action(self, action_idx: int) -> tuple:
        """Decode flat action index into (transform_type, intensity_level).

        Transform types: 0=rotation, 1=brightness, 2=contrast, 3=noise, 4=blur
        Intensity levels: 0-11 (12 levels)
        """
        transform_type = action_idx // 12
        intensity_level = action_idx % 12
        return transform_type, intensity_level

    def store_transition(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: float,
        value: torch.Tensor,
        done: bool,
    ):
        self.buffer["states"].append(state)
        self.buffer["actions"].append(action)
        self.buffer["log_probs"].append(log_prob)
        self.buffer["rewards"].append(reward)
        self.buffer["values"].append(value)
        self.buffer["dones"].append(done)

    def compute_gae(self, next_value: torch.Tensor) -> tuple:
        """Compute Generalized Advantage Estimation (GAE)."""
        rewards = self.buffer["rewards"]
        values = self.buffer["values"]
        dones = self.buffer["dones"]

        T = len(rewards)
        advantages = torch.zeros(T, device=next_value.device)
        returns = torch.zeros(T, device=next_value.device)

        gae = 0.0
        for t in reversed(range(T)):
            if t == T - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = gae + values[t]

        return advantages, returns

    def update(self, next_value: torch.Tensor, n_epochs: int = 4) -> dict:
        """Run PPO update using collected experience.

        Returns dict with actor_loss, critic_loss, entropy, approx_kl.
        """
        advantages, returns = self.compute_gae(next_value)

        states = torch.stack(self.buffer["states"])
        actions = torch.stack(self.buffer["actions"])
        old_log_probs = torch.stack(self.buffer["log_probs"])

        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        n_updates = 0

        for _ in range(n_epochs):
            indices = torch.randperm(len(states))
            for start in range(0, len(states), self.batch_size):
                end = min(start + self.batch_size, len(states))
                idx = indices[start:end]

                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_advantages = advantages[idx]
                batch_returns = returns[idx]

                action_probs, values = self.forward(batch_states)
                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # PPO clipped surrogate
                ratio = (new_log_probs - batch_old_log_probs).exp()
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                    * batch_advantages
                )
                actor_loss = -torch.min(surr1, surr2).mean()
                actor_loss = actor_loss - self.entropy_coeff * entropy

                # value loss
                critic_loss = F.mse_loss(values, batch_returns)

                # update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(
                    list(self.trunk.parameters()) + list(self.policy_head.parameters()),
                    max_norm=0.5,
                )
                self.actor_optimizer.step()

                # update critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.trunk.parameters()) + list(self.value_head.parameters()),
                    max_norm=0.5,
                )
                self.critic_optimizer.step()

                with torch.no_grad():
                    approx_kl = (batch_old_log_probs - new_log_probs).mean()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()
                total_kl += approx_kl.item()
                n_updates += 1

        self._reset_buffer()

        return {
            "actor_loss": total_actor_loss / max(n_updates, 1),
            "critic_loss": total_critic_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
            "approx_kl": total_kl / max(n_updates, 1),
        }

    def get_policy_entropy(self, states: torch.Tensor) -> float:
        """Compute policy entropy H(π) over a batch of states."""
        with torch.no_grad():
            action_probs, _ = self.forward(states)
            dist = Categorical(action_probs)
            return dist.entropy().mean().item()

    def get_action_distribution(self, states: torch.Tensor) -> torch.Tensor:
        """Return mean action probabilities across a batch of states."""
        with torch.no_grad():
            action_probs, _ = self.forward(states)
            return action_probs.mean(dim=0)
