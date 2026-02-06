"""
Entry point for evaluation on internal test set and external validation cohorts.

Usage:
    # Internal test set
    python scripts/evaluate.py --config configs/default.yaml \
        --checkpoint outputs/seed_42/best_model.pt --split test

    # External validation
    python scripts/evaluate.py --config configs/default.yaml \
        --checkpoint outputs/seed_42/best_model.pt --external chestxray14

    # All external cohorts
    python scripts/evaluate.py --config configs/default.yaml \
        --checkpoint outputs/seed_42/best_model.pt --external all
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.cnn_transformer import CNNTransformerClassifier
from src.data.datasets import get_combined_dataloader, create_patient_splits, ExternalDataset
from src.data.preprocessing import get_eval_transforms
from src.training.evaluator import Evaluator
from src.utils.logging_utils import load_checkpoint
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--external", type=str, default=None,
                        help="External dataset: chestxray14, vindrcxr, kermany, or all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--save_predictions", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    classifier = CNNTransformerClassifier(
        in_channels=3,
        num_classes=2,
        d_model=config.get("classifier", {}).get("d_model", 64),
        nhead=config.get("classifier", {}).get("nhead", 4),
        dim_ff=config.get("classifier", {}).get("dim_ff", 256),
        n_layers=config.get("classifier", {}).get("n_layers", 2),
        dropout=config.get("classifier", {}).get("dropout", 0.3),
        use_mixstyle=config.get("classifier", {}).get("use_mixstyle", False),
    ).to(device)

    ckpt = load_checkpoint(args.checkpoint, device=device)
    classifier.load_state_dict(ckpt["classifier_state_dict"])
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    evaluator = Evaluator(classifier, device, seed=args.seed)
    data_root = config.get("data_root", "/data")
    eval_transform = get_eval_transforms()
    batch_size = config.get("classifier", {}).get("batch_size", 64)

    os.makedirs(args.output_dir, exist_ok=True)

    # internal evaluation
    if args.external is None:
        splits = create_patient_splits(data_root, data_root, seed=args.seed)
        patient_ids = splits[f"{args.split}_patients"]

        loader = get_combined_dataloader(
            data_root, split=args.split, batch_size=batch_size,
            num_workers=4, transform=eval_transform,
            patient_ids=patient_ids, shuffle=False,
        )

        metrics = evaluator.evaluate(loader, name=f"internal_{args.split}")
        _print_metrics(metrics, f"Internal {args.split}")

        # save
        out_path = os.path.join(args.output_dir, f"internal_{args.split}_seed{args.seed}.json")
        _save_metrics(metrics, out_path)

        if args.save_predictions:
            evaluator.save_predictions(
                loader, os.path.join(args.output_dir, "predictions"),
                split=args.split,
            )

    # external evaluation
    else:
        datasets_to_eval = []
        if args.external == "all":
            datasets_to_eval = ["chestxray14", "vindrcxr", "kermany"]
        else:
            datasets_to_eval = [args.external]

        external_loaders = {}
        for ds_name in datasets_to_eval:
            ds = ExternalDataset(data_root, ds_name, transform=eval_transform)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
            external_loaders[ds_name] = loader

        # get internal AUC for delta computation
        internal_auc = ckpt.get("metrics", {}).get("auc", 0.981)

        results = evaluator.evaluate_external(external_loaders, internal_auc)

        for name, metrics in results.items():
            if name != "average":
                _print_metrics(metrics, f"External: {name}")

        if "average" in results:
            print(f"\n--- External Average ---")
            avg = results["average"]
            print(f"  AUC:       {avg['auc']:.4f}")
            print(f"  ECE:       {avg['ece']:.4f}")
            print(f"  Δ AUC:     {avg['delta_auc']:.4f}")
            print(f"  Accuracy:  {avg['accuracy']:.4f}")

        # save
        out_path = os.path.join(args.output_dir, f"external_seed{args.seed}.json")
        _save_metrics(results, out_path)

        if args.save_predictions:
            for ds_name, loader in external_loaders.items():
                evaluator.save_predictions(
                    loader, os.path.join(args.output_dir, "predictions"),
                    split=ds_name,
                )


def _print_metrics(metrics: dict, title: str):
    print(f"\n--- {title} (n={metrics.get('n_samples', '?')}) ---")
    print(f"  Accuracy:    {metrics['accuracy']:.4f} "
          f"({metrics.get('acc_ci_lower', 0):.4f}-{metrics.get('acc_ci_upper', 0):.4f})")
    print(f"  AUC:         {metrics['auc']:.4f} "
          f"({metrics.get('auc_ci_lower', 0):.4f}-{metrics.get('auc_ci_upper', 0):.4f})")
    print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  ECE:         {metrics['ece']:.4f}")
    print(f"  Brier:       {metrics.get('brier', 0):.4f}")
    if "ppv_at_5pct" in metrics:
        print(f"  PPV @5%:     {metrics['ppv_at_5pct']:.4f}")
        print(f"  NPV @5%:     {metrics['npv_at_5pct']:.4f}")
    if "delta_auc" in metrics:
        print(f"  Δ AUC:       {metrics['delta_auc']:.4f}")


def _save_metrics(metrics: dict, path: str):
    """Save metrics as JSON, handling numpy types."""
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(path, "w") as f:
        json.dump(convert(metrics), f, indent=2)
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
