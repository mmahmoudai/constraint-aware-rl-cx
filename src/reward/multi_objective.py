"""
Multi-objective reward function for the PPO augmentation agent.

R_t = α·Δ_acc + β·D_aug + δ·Δ_cal - γ_c·C_comp - ε·P_implaus

Components:
    Δ_acc:     AUC_current - AUC_baseline
    D_aug:     augmentation diversity H(π) = -Σ p_k log p_k
    Δ_cal:     ECE_baseline - ECE_current (positive when calibration improves)
    C_comp:    T_aug / T_baseline (computational cost ratio)
    P_implaus: max(0, τ_clinical - S_clinical) (implausibility penalty)

Default weights: α=1.0, β=0.3, δ=0.5, γ_c=0.1, ε=2.0
"""

import math
from typing import Optional

import torch
import numpy as np


class MultiObjectiveReward:
    """Computes the multi-objective reward signal for the PPO agent.

    Args:
        alpha:         weight for accuracy improvement (default 1.0)
        beta:          weight for augmentation diversity (default 0.3)
        delta:         weight for calibration improvement (default 0.5)
        gamma_cost:    weight for computational cost (default 0.1)
        epsilon:       weight for implausibility penalty (default 2.0)
        baseline_auc:  initial AUC before augmentation (updated online)
        baseline_ece:  initial ECE before augmentation (updated online)
        baseline_time: baseline augmentation time (seconds)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.3,
        delta: float = 0.5,
        gamma_cost: float = 0.1,
        epsilon: float = 2.0,
    ):
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.gamma_cost = gamma_cost
        self.epsilon = epsilon

        # running baselines (updated after each evaluation)
        self.baseline_auc = 0.5
        self.baseline_ece = 0.1
        self.baseline_time = 1.0

        # action frequency tracker for diversity computation
        self.action_counts = np.zeros(60)
        self.total_actions = 0

    def compute(
        self,
        current_auc: float,
        current_ece: float,
        action_idx: int,
        aug_time: float = 1.0,
        implausibility: float = 0.0,
    ) -> dict:
        """Compute the full reward signal.

        Args:
            current_auc:    AUC after applying augmentation
            current_ece:    ECE after applying augmentation
            action_idx:     selected action index (for diversity tracking)
            aug_time:       wall-clock time for augmentation (seconds)
            implausibility: clinical implausibility score from ClinicalTransforms

        Returns:
            dict with total reward and component breakdown.
        """
        # accuracy improvement
        delta_acc = current_auc - self.baseline_auc

        # augmentation diversity (entropy of action distribution)
        self._update_action_counts(action_idx)
        diversity = self._compute_diversity()

        # calibration improvement (positive = better calibration)
        delta_cal = self.baseline_ece - current_ece

        # computational cost ratio
        cost_ratio = aug_time / max(self.baseline_time, 1e-6)

        # implausibility penalty
        penalty = max(0.0, implausibility)

        # total reward
        total = (
            self.alpha * delta_acc
            + self.beta * diversity
            + self.delta * delta_cal
            - self.gamma_cost * cost_ratio
            - self.epsilon * penalty
        )

        return {
            "total": total,
            "delta_acc": delta_acc,
            "diversity": diversity,
            "delta_cal": delta_cal,
            "cost_ratio": cost_ratio,
            "implausibility": penalty,
            "components": {
                "alpha_delta_acc": self.alpha * delta_acc,
                "beta_diversity": self.beta * diversity,
                "delta_delta_cal": self.delta * delta_cal,
                "gamma_cost": -self.gamma_cost * cost_ratio,
                "epsilon_implaus": -self.epsilon * penalty,
            },
        }

    def _update_action_counts(self, action_idx: int):
        """Track action frequency for diversity computation."""
        self.action_counts[action_idx] += 1
        self.total_actions += 1

    def _compute_diversity(self) -> float:
        """Compute Shannon entropy of the action distribution.

        H(π) = -Σ p_k log p_k

        Uniform over 60 actions gives H_max = ln(60) ≈ 4.09 nats.
        """
        if self.total_actions == 0:
            return 0.0

        probs = self.action_counts / self.total_actions
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log(probs))
        return float(entropy)

    def update_baselines(self, auc: float, ece: float, aug_time: float = 1.0):
        """Update baseline values after evaluation."""
        self.baseline_auc = auc
        self.baseline_ece = ece
        self.baseline_time = aug_time

    def reset_action_counts(self):
        """Reset action frequency tracker (e.g., at start of each epoch)."""
        self.action_counts = np.zeros(60)
        self.total_actions = 0

    def get_action_distribution(self) -> np.ndarray:
        """Return current action probability distribution."""
        if self.total_actions == 0:
            return np.ones(60) / 60
        return self.action_counts / self.total_actions

    def state_dict(self) -> dict:
        return {
            "baseline_auc": self.baseline_auc,
            "baseline_ece": self.baseline_ece,
            "baseline_time": self.baseline_time,
            "action_counts": self.action_counts.copy(),
            "total_actions": self.total_actions,
            "weights": {
                "alpha": self.alpha,
                "beta": self.beta,
                "delta": self.delta,
                "gamma_cost": self.gamma_cost,
                "epsilon": self.epsilon,
            },
        }

    def load_state_dict(self, state: dict):
        self.baseline_auc = state["baseline_auc"]
        self.baseline_ece = state["baseline_ece"]
        self.baseline_time = state["baseline_time"]
        self.action_counts = state["action_counts"]
        self.total_actions = state["total_actions"]
