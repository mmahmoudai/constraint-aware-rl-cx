"""
Calibration utilities: temperature scaling, reliability diagrams, Brier score.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize_scalar


class TemperatureScaling(nn.Module):
    """Post-hoc temperature scaling for probability calibration.

    Learns a single scalar T such that calibrated probabilities are:
        p_cal = softmax(logits / T)

    Optimized on a held-out validation set by minimizing NLL.
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 100,
    ):
        """Fit temperature on validation logits and labels.

        Args:
            logits: (N, C) raw logits from the model
            labels: (N,) integer class labels
            lr:     learning rate for LBFGS
            max_iter: maximum optimization iterations
        """
        self.temperature.data = torch.ones(1, device=logits.device)
        nll = nn.CrossEntropyLoss()

        optimizer = torch.optim.LBFGS(
            [self.temperature], lr=lr, max_iter=max_iter
        )

        def closure():
            optimizer.zero_grad()
            loss = nll(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return self.temperature.item()

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply learned temperature to produce calibrated probabilities."""
        with torch.no_grad():
            return F.softmax(self.forward(logits), dim=-1)


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "equal_frequency",
) -> float:
    """Compute Expected Calibration Error.

    Args:
        y_true:   binary labels (N,)
        y_prob:   predicted probability for class 1 (N,)
        n_bins:   number of bins (default 10)
        strategy: 'equal_frequency' or 'equal_width'

    Returns:
        ECE value
    """
    n = len(y_true)
    if n == 0:
        return 0.0

    if strategy == "equal_frequency":
        sorted_idx = np.argsort(y_prob)
        bin_size = n // n_bins
        ece = 0.0
        for i in range(n_bins):
            start = i * bin_size
            end = start + bin_size if i < n_bins - 1 else n
            idx = sorted_idx[start:end]
            if len(idx) == 0:
                continue
            bin_acc = y_true[idx].mean()
            bin_conf = y_prob[idx].mean()
            ece += len(idx) / n * abs(bin_acc - bin_conf)
        return ece

    else:  # equal_width
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (y_prob >= lo) & (y_prob < hi)
            if not mask.any():
                continue
            bin_acc = y_true[mask].mean()
            bin_conf = y_prob[mask].mean()
            ece += mask.sum() / n * abs(bin_acc - bin_conf)
        return ece


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Brier score: mean squared error of predicted probabilities."""
    return float(np.mean((y_prob - y_true) ** 2))


def reliability_diagram_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Compute data for a reliability diagram.

    Returns:
        dict with bin_centers, bin_accuracies, bin_confidences, bin_counts
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    centers = []
    accuracies = []
    confidences = []
    counts = []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        count = mask.sum()
        centers.append((lo + hi) / 2)
        counts.append(int(count))
        if count > 0:
            accuracies.append(float(y_true[mask].mean()))
            confidences.append(float(y_prob[mask].mean()))
        else:
            accuracies.append(0.0)
            confidences.append((lo + hi) / 2)

    return {
        "bin_centers": centers,
        "bin_accuracies": accuracies,
        "bin_confidences": confidences,
        "bin_counts": counts,
    }
