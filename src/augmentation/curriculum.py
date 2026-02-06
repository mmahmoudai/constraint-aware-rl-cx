"""
Curriculum scheduler with safety rollback for augmentation intensity.

Five stages with linearly spaced difficulty: {0, 0.25, 0.5, 0.75, 1.0}.
Rollback fires when ECE increases by more than τ_rollback = 0.02 between stages.
Within each stage, augmentation weight ramps as:
    λ_aug(t) = d_k · (1 - exp(-t / τ_adapt))  with τ_adapt = 100
"""

import math
from typing import Optional


class CurriculumScheduler:
    """Manages progressive difficulty scheduling with ECE-based safety rollback.

    Args:
        n_stages:        number of curriculum stages (default 5)
        rollback_thresh: ECE increase threshold triggering rollback (default 0.02)
        tau_adapt:       exponential ramp time constant (default 100)
        epochs_per_stage: training epochs per curriculum stage
    """

    def __init__(
        self,
        n_stages: int = 5,
        rollback_thresh: float = 0.02,
        tau_adapt: float = 100.0,
        total_epochs: int = 150,
    ):
        self.n_stages = n_stages
        self.rollback_thresh = rollback_thresh
        self.tau_adapt = tau_adapt
        self.total_epochs = total_epochs
        self.epochs_per_stage = total_epochs // n_stages

        # difficulty levels: {0, 0.25, 0.5, 0.75, 1.0}
        self.difficulty_levels = [
            (k) / (n_stages - 1) for k in range(n_stages)
        ]

        self.current_stage = 0
        self.step_in_stage = 0
        self.prev_ece = None
        self.rollback_count = 0
        self.rollback_history = []

    @property
    def difficulty(self) -> float:
        """Current curriculum difficulty level."""
        return self.difficulty_levels[self.current_stage]

    @property
    def aug_weight(self) -> float:
        """Current augmentation weight with exponential ramp."""
        d_k = self.difficulty
        t = self.step_in_stage
        return d_k * (1.0 - math.exp(-t / self.tau_adapt))

    def should_advance(self, epoch: int) -> bool:
        """Check if it's time to advance to the next stage."""
        stage_for_epoch = min(epoch // self.epochs_per_stage, self.n_stages - 1)
        return stage_for_epoch > self.current_stage

    def try_advance(self, current_ece: float, epoch: int) -> dict:
        """Attempt to advance to the next curriculum stage.

        If ECE increased by more than rollback_thresh, revert instead.

        Args:
            current_ece: ECE measured at end of current stage
            epoch: current training epoch

        Returns:
            dict with keys: advanced (bool), rolled_back (bool),
                           stage (int), difficulty (float), ece_delta (float)
        """
        result = {
            "advanced": False,
            "rolled_back": False,
            "stage": self.current_stage,
            "difficulty": self.difficulty,
            "ece_delta": 0.0,
        }

        if not self.should_advance(epoch):
            return result

        if self.current_stage >= self.n_stages - 1:
            return result

        ece_delta = 0.0
        if self.prev_ece is not None:
            ece_delta = current_ece - self.prev_ece

        result["ece_delta"] = ece_delta

        # check rollback condition
        if self.prev_ece is not None and ece_delta > self.rollback_thresh:
            # rollback: revert to previous stage
            if self.current_stage > 0:
                old_stage = self.current_stage
                self.current_stage -= 1
                self.step_in_stage = 0
                self.rollback_count += 1
                self.rollback_history.append({
                    "epoch": epoch,
                    "from_stage": old_stage,
                    "to_stage": self.current_stage,
                    "ece_delta": ece_delta,
                })
                result["rolled_back"] = True
                result["stage"] = self.current_stage
                result["difficulty"] = self.difficulty
            return result

        # advance
        self.current_stage += 1
        self.step_in_stage = 0
        self.prev_ece = current_ece
        result["advanced"] = True
        result["stage"] = self.current_stage
        result["difficulty"] = self.difficulty
        return result

    def step(self):
        """Increment step counter within current stage."""
        self.step_in_stage += 1

    def update_ece(self, ece: float):
        """Update stored ECE for rollback comparison."""
        self.prev_ece = ece

    def state_dict(self) -> dict:
        return {
            "current_stage": self.current_stage,
            "step_in_stage": self.step_in_stage,
            "prev_ece": self.prev_ece,
            "rollback_count": self.rollback_count,
            "rollback_history": self.rollback_history,
        }

    def load_state_dict(self, state: dict):
        self.current_stage = state["current_stage"]
        self.step_in_stage = state["step_in_stage"]
        self.prev_ece = state["prev_ece"]
        self.rollback_count = state["rollback_count"]
        self.rollback_history = state["rollback_history"]


class RandAugCurriculumScheduler:
    """Curriculum scheduler for RandAugment baseline.

    Linearly increases magnitude M from m_start to m_end across stages.
    """

    def __init__(
        self,
        n_stages: int = 5,
        m_start: int = 3,
        m_end: int = 12,
        total_epochs: int = 150,
    ):
        self.n_stages = n_stages
        self.m_start = m_start
        self.m_end = m_end
        self.total_epochs = total_epochs
        self.epochs_per_stage = total_epochs // n_stages

    def get_magnitude(self, epoch: int) -> int:
        """Return RandAugment magnitude M for the given epoch."""
        stage = min(epoch // self.epochs_per_stage, self.n_stages - 1)
        fraction = stage / max(self.n_stages - 1, 1)
        m = self.m_start + fraction * (self.m_end - self.m_start)
        return int(round(m))
