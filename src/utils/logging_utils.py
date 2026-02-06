"""
Logging utilities: TensorBoard integration and CSV logging.
"""

import os
import csv
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import torch


def setup_logger(
    name: str,
    log_dir: str,
    level: int = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """Set up a logger with file and optional console handlers.

    Args:
        name:    logger name
        log_dir: directory for log files
        level:   logging level
        console: whether to also log to console
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    if console:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


class TBLogger:
    """TensorBoard + CSV logger for experiment tracking.

    Logs scalar metrics to TensorBoard and saves them as CSV for offline analysis.
    """

    def __init__(self, log_dir: str, experiment_name: str = "default"):
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._tb_writer = None
        self._csv_path = self.log_dir / "metrics.csv"
        self._csv_initialized = False
        self._all_keys = set()
        self._rows = []

    @property
    def tb_writer(self):
        if self._tb_writer is None:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb_writer = SummaryWriter(str(self.log_dir / "tb"))
            except ImportError:
                self._tb_writer = None
        return self._tb_writer

    def log_scalar(self, tag: str, value: float, step: int):
        """Log a single scalar value."""
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(tag, value, step)

    def log_scalars(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        """Log multiple scalar values and append to CSV."""
        row = {"step": step}
        for key, value in metrics.items():
            tag = f"{prefix}/{key}" if prefix else key
            self.log_scalar(tag, value, step)
            row[tag] = value

        self._all_keys.update(row.keys())
        self._rows.append(row)

    def log_config(self, config: dict):
        """Save experiment configuration as JSON."""
        config_path = self.log_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)

    def save_csv(self):
        """Write accumulated rows to CSV."""
        if not self._rows:
            return

        keys = sorted(self._all_keys)
        with open(self._csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            for row in self._rows:
                writer.writerow(row)

    def save_predictions(
        self,
        y_true,
        y_prob,
        y_pred,
        split: str = "test",
        seed: int = 42,
    ):
        """Save per-sample predictions for reproducibility."""
        import numpy as np

        pred_dir = self.log_dir / "predictions"
        pred_dir.mkdir(exist_ok=True)

        np.savez(
            pred_dir / f"{split}_seed{seed}.npz",
            y_true=np.asarray(y_true),
            y_prob=np.asarray(y_prob),
            y_pred=np.asarray(y_pred),
        )

    def close(self):
        """Flush and close all writers."""
        self.save_csv()
        if self._tb_writer is not None:
            self._tb_writer.close()


def save_checkpoint(
    path: str,
    epoch: int,
    classifier,
    agent=None,
    optimizer=None,
    scheduler=None,
    curriculum=None,
    metrics: Optional[dict] = None,
    config: Optional[dict] = None,
):
    """Save a training checkpoint."""
    state = {
        "epoch": epoch,
        "classifier_state_dict": classifier.state_dict(),
    }
    if agent is not None:
        state["agent_state_dict"] = agent.state_dict()
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    if curriculum is not None:
        state["curriculum_state_dict"] = curriculum.state_dict()
    if metrics is not None:
        state["metrics"] = metrics
    if config is not None:
        state["config"] = config

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, device: torch.device = torch.device("cpu")) -> dict:
    """Load a training checkpoint."""
    return torch.load(path, map_location=device)
