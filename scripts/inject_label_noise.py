"""
Label-noise stress test: inject symmetric noise at rates η ∈ {0, 5, 10, 15, 20}%
into training labels and retrain to assess robustness.

Usage:
    python scripts/inject_label_noise.py --config configs/default.yaml \
        --noise_rates 0.0 0.05 0.10 0.15 0.20 --seeds 42 123 2025
"""

import argparse
import copy
import json
import os
import sys
import random

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.datasets import get_combined_dataloader, create_patient_splits
from src.data.preprocessing import get_train_transforms, get_eval_transforms
from src.training.trainer import Trainer
from src.training.evaluator import Evaluator
from src.models.cnn_transformer import CNNTransformerClassifier
from src.utils.logging_utils import load_checkpoint


class NoisyLabelWrapper(torch.utils.data.Dataset):
    """Wraps a dataset and flips a fraction η of labels randomly."""

    def __init__(self, dataset, noise_rate: float, seed: int = 42):
        self.dataset = dataset
        self.noise_rate = noise_rate
        rng = np.random.RandomState(seed)

        n = len(dataset)
        self.flip_mask = rng.random(n) < noise_rate
        self.flipped_labels = {}

        for i in range(n):
            if self.flip_mask[i]:
                _, original_label = dataset[i]
                self.flipped_labels[i] = 1 - original_label  # binary flip

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if idx in self.flipped_labels:
            label = self.flipped_labels[idx]
        return image, label


def main():
    parser = argparse.ArgumentParser(description="Label-noise stress test")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--noise_rates", type=float, nargs="+",
                        default=[0.0, 0.05, 0.10, 0.15, 0.20])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2025])
    parser.add_argument("--output_dir", type=str, default="outputs/label_noise")
    args = parser.parse_args()

    with open(args.config) as f:
        base_config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = base_config.get("data_root", "/data")
    eval_transform = get_eval_transforms()
    batch_size = base_config.get("classifier", {}).get("batch_size", 64)

    results = {}

    for noise_rate in args.noise_rates:
        results[f"eta_{noise_rate}"] = {}

        for seed in args.seeds:
            print(f"\n{'='*60}")
            print(f"Noise rate: {noise_rate:.0%} | Seed: {seed}")
            print(f"{'='*60}")

            # set seed
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            # create splits
            splits = create_patient_splits(data_root, data_root, seed=seed)

            # build train loader with noisy labels
            train_transform = get_train_transforms()
            train_loader = get_combined_dataloader(
                data_root, split="train", batch_size=batch_size,
                num_workers=4, transform=train_transform,
                patient_ids=splits["train_patients"],
            )

            if noise_rate > 0:
                noisy_dataset = NoisyLabelWrapper(
                    train_loader.dataset, noise_rate=noise_rate, seed=seed
                )
                train_loader = torch.utils.data.DataLoader(
                    noisy_dataset, batch_size=batch_size, shuffle=True,
                    num_workers=4, pin_memory=True, drop_last=True,
                )

            val_loader = get_combined_dataloader(
                data_root, split="val", batch_size=batch_size,
                num_workers=4, transform=eval_transform,
                patient_ids=splits["val_patients"], shuffle=False,
            )
            test_loader = get_combined_dataloader(
                data_root, split="test", batch_size=batch_size,
                num_workers=4, transform=eval_transform,
                patient_ids=splits["test_patients"], shuffle=False,
            )

            # train
            config = copy.deepcopy(base_config)
            config["seed"] = seed
            config["output_dir"] = os.path.join(
                args.output_dir, f"eta_{noise_rate:.2f}"
            )
            config["experiment_name"] = f"seed_{seed}"

            trainer = Trainer(
                config=config,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
            )
            trainer.train()

            # evaluate on clean test set
            ckpt_path = os.path.join(
                config["output_dir"], f"seed_{seed}", "best_model.pt"
            )
            if os.path.exists(ckpt_path):
                ckpt = load_checkpoint(ckpt_path, device=device)
                classifier = CNNTransformerClassifier(
                    **{k: config.get("classifier", {}).get(k, v)
                       for k, v in [("d_model", 64), ("nhead", 4),
                                    ("dim_ff", 256), ("n_layers", 2),
                                    ("dropout", 0.3)]}
                ).to(device)
                classifier.load_state_dict(ckpt["classifier_state_dict"])

                evaluator = Evaluator(classifier, device, seed=seed)
                metrics = evaluator.evaluate(test_loader, name="test")

                results[f"eta_{noise_rate}"][f"seed_{seed}"] = {
                    "accuracy": metrics["accuracy"],
                    "auc": metrics["auc"],
                    "ece": metrics["ece"],
                }

                print(f"  Acc: {metrics['accuracy']:.4f} | "
                      f"AUC: {metrics['auc']:.4f} | "
                      f"ECE: {metrics['ece']:.4f}")

    # save summary
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output_dir}/summary.json")


if __name__ == "__main__":
    main()
