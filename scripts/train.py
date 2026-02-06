"""
Entry point for training the constraint-aware RL augmentation framework.

Usage:
    # Single GPU
    python scripts/train.py --config configs/default.yaml --seed 42

    # Multi-GPU (4× A100)
    torchrun --nproc_per_node=4 scripts/train.py --config configs/default.yaml --seed 42
"""

import argparse
import os
import sys
import random

import numpy as np
import torch
import torch.distributed as dist
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.datasets import get_combined_dataloader, create_patient_splits, ExternalDataset
from src.data.preprocessing import get_train_transforms, get_eval_transforms
from src.augmentation.clinical_transforms import (
    ClinicalTransforms,
    RandomAugmentation,
    RandAugmentCXR,
    TrivialAugmentCXR,
)
from src.training.trainer import Trainer


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_augment_fn(config: dict):
    """Build the augmentation function based on config."""
    aug_type = config.get("augmentation", {}).get("type", "rl")

    if aug_type == "none":
        return None
    elif aug_type == "random":
        return RandomAugmentation(difficulty=1.0)
    elif aug_type == "randaugment":
        n = config["augmentation"].get("rand_n", 2)
        m = config["augmentation"].get("rand_m", 9)
        return RandAugmentCXR(n=n, m=m)
    elif aug_type == "trivialaugment":
        return TrivialAugmentCXR()
    elif aug_type == "rl":
        return None  # handled internally by Trainer
    else:
        return None


def main():
    parser = argparse.ArgumentParser(description="Train constraint-aware RL augmentation")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed")
    parser.add_argument("--local_rank", type=int, default=0, help="DDP local rank")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.seed is not None:
        config["seed"] = args.seed

    seed = config.get("seed", 42)
    set_seed(seed)

    # DDP setup
    distributed = "LOCAL_RANK" in os.environ
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        rank = dist.get_rank()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rank = 0

    if rank == 0:
        print(f"Config: {args.config}")
        print(f"Seed: {seed}")
        print(f"Device: {device}")
        print(f"Distributed: {distributed}")

    # create patient-level splits
    data_root = config.get("data_root", "/data")
    splits = create_patient_splits(data_root, data_root, seed=seed)

    # build augmentation function for non-RL baselines
    augment_fn = get_augment_fn(config)

    # build data loaders
    train_transform = get_train_transforms(augment_fn=augment_fn)
    eval_transform = get_eval_transforms()

    batch_size = config.get("classifier", {}).get("batch_size", 64)
    num_workers = config.get("num_workers", 8)

    train_loader = get_combined_dataloader(
        data_root, split="train", batch_size=batch_size,
        num_workers=num_workers, transform=train_transform,
        patient_ids=splits["train_patients"],
    )
    val_loader = get_combined_dataloader(
        data_root, split="val", batch_size=batch_size,
        num_workers=num_workers, transform=eval_transform,
        patient_ids=splits["val_patients"], shuffle=False,
    )

    # update config with output directory
    experiment_name = config.get("experiment_name", "default")
    config["output_dir"] = os.path.join(
        config.get("output_dir", "outputs"), experiment_name
    )
    config["experiment_name"] = f"seed_{seed}"

    # train
    trainer = Trainer(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        rank=rank,
    )
    trainer.train()

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
