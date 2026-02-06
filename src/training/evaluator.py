"""
Evaluation module: internal test set, external validation, calibration analysis,
statistical tests (McNemar, DeLong, Cohen's d), and per-seed prediction saving.
"""

import os
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..models.cnn_transformer import CNNTransformerClassifier
from ..utils.metrics import (
    compute_metrics,
    expected_calibration_error,
    mcnemar_test,
    delong_test,
    cohens_d,
    prevalence_adjusted_ppv_npv,
    bootstrap_ci,
)
from ..utils.calibration import (
    TemperatureScaling,
    brier_score,
    reliability_diagram_data,
)


class Evaluator:
    """Comprehensive evaluation of trained classifiers.

    Supports internal test set evaluation, external validation on multiple cohorts,
    calibration analysis, statistical comparisons, and prevalence-adjusted metrics.

    Args:
        classifier: trained CNNTransformerClassifier
        device:     torch device
        seed:       random seed for reproducibility
    """

    def __init__(
        self,
        classifier: CNNTransformerClassifier,
        device: torch.device,
        seed: int = 42,
    ):
        self.classifier = classifier
        self.device = device
        self.seed = seed

    @torch.no_grad()
    def predict(self, dataloader: DataLoader) -> dict:
        """Run inference on a dataloader.

        Returns dict with y_true, y_prob, y_pred, logits.
        """
        self.classifier.eval()
        all_logits = []
        all_probs = []
        all_labels = []

        for images, labels in dataloader:
            images = images.to(self.device)
            logits = self.classifier(images)
            probs = F.softmax(logits, dim=-1)

            all_logits.append(logits.cpu())
            all_probs.append(probs[:, 1].cpu())
            all_labels.append(labels)

        logits = torch.cat(all_logits).numpy()
        y_prob = torch.cat(all_probs).numpy()
        y_true = torch.cat(all_labels).numpy()
        y_pred = (y_prob >= 0.5).astype(int)

        return {
            "y_true": y_true,
            "y_prob": y_prob,
            "y_pred": y_pred,
            "logits": logits,
        }

    def evaluate(self, dataloader: DataLoader, name: str = "test") -> dict:
        """Full evaluation: metrics + ECE + Brier + confidence intervals.

        Args:
            dataloader: evaluation DataLoader
            name: dataset name for reporting

        Returns:
            dict with all metrics
        """
        preds = self.predict(dataloader)
        y_true, y_prob, y_pred = preds["y_true"], preds["y_prob"], preds["y_pred"]

        metrics = compute_metrics(y_true, y_prob)
        metrics["ece"] = expected_calibration_error(y_true, y_prob, n_bins=10)
        metrics["brier"] = brier_score(y_true, y_prob)
        metrics["n_samples"] = len(y_true)
        metrics["dataset"] = name

        # bootstrap 95% CI for AUC
        from sklearn.metrics import roc_auc_score
        auc_lo, auc_hi = bootstrap_ci(
            y_true, y_prob, roc_auc_score, n_bootstrap=1000, seed=self.seed
        )
        metrics["auc_ci_lower"] = auc_lo
        metrics["auc_ci_upper"] = auc_hi

        # bootstrap 95% CI for accuracy
        acc_fn = lambda yt, yp: (yp >= 0.5).astype(int).mean() if False else np.mean((yp >= 0.5) == yt)
        acc_lo, acc_hi = bootstrap_ci(
            y_true, y_prob, acc_fn, n_bootstrap=1000, seed=self.seed
        )
        metrics["acc_ci_lower"] = acc_lo
        metrics["acc_ci_upper"] = acc_hi

        # prevalence-adjusted PPV/NPV at 5%
        prev_metrics = prevalence_adjusted_ppv_npv(
            metrics["sensitivity"], metrics["specificity"], prevalence=0.05
        )
        metrics["ppv_at_5pct"] = prev_metrics["ppv"]
        metrics["npv_at_5pct"] = prev_metrics["npv"]

        # reliability diagram data
        metrics["reliability"] = reliability_diagram_data(y_true, y_prob)

        return metrics

    def evaluate_external(
        self,
        external_loaders: dict,
        internal_auc: float,
    ) -> dict:
        """Evaluate on external validation cohorts and compute AUC degradation.

        Args:
            external_loaders: dict mapping dataset_name -> DataLoader
            internal_auc: AUC on internal test set (for computing delta)

        Returns:
            dict mapping dataset_name -> metrics
        """
        results = {}
        for name, loader in external_loaders.items():
            metrics = self.evaluate(loader, name=name)
            metrics["internal_auc"] = internal_auc
            metrics["delta_auc"] = metrics["auc"] - internal_auc
            results[name] = metrics

        # compute averages
        if results:
            avg = {
                "auc": np.mean([r["auc"] for r in results.values()]),
                "ece": np.mean([r["ece"] for r in results.values()]),
                "accuracy": np.mean([r["accuracy"] for r in results.values()]),
                "delta_auc": np.mean([r["delta_auc"] for r in results.values()]),
            }
            results["average"] = avg

        return results

    def compare_methods(
        self,
        dataloader: DataLoader,
        method_checkpoints: dict,
    ) -> dict:
        """Run pairwise statistical comparisons between methods.

        Args:
            dataloader: shared test DataLoader
            method_checkpoints: dict mapping method_name -> checkpoint_path

        Returns:
            dict with pairwise McNemar, DeLong, Cohen's d results
        """
        # collect predictions from all methods
        predictions = {}
        for name, ckpt_path in method_checkpoints.items():
            state = torch.load(ckpt_path, map_location=self.device)
            self.classifier.load_state_dict(state["classifier_state_dict"])
            preds = self.predict(dataloader)
            predictions[name] = preds

        # pairwise comparisons
        methods = list(predictions.keys())
        comparisons = {}

        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                a, b = methods[i], methods[j]
                pa, pb = predictions[a], predictions[b]
                key = f"{a}_vs_{b}"

                comparisons[key] = {
                    "mcnemar": mcnemar_test(
                        pa["y_true"], pa["y_pred"], pb["y_pred"]
                    ),
                    "delong": delong_test(
                        pa["y_true"], pa["y_prob"], pb["y_prob"]
                    ),
                    "cohens_d": cohens_d(
                        (pa["y_pred"] == pa["y_true"]).astype(float),
                        (pb["y_pred"] == pb["y_true"]).astype(float),
                    ),
                }

        return comparisons

    def calibration_analysis(
        self,
        val_loader: DataLoader,
        test_loader: DataLoader,
    ) -> dict:
        """Run calibration analysis including temperature scaling comparison.

        Args:
            val_loader: validation set for fitting temperature
            test_loader: test set for evaluation

        Returns:
            dict with pre/post calibration metrics
        """
        # get validation logits for temperature fitting
        val_preds = self.predict(val_loader)
        test_preds = self.predict(test_loader)

        # pre-calibration ECE
        pre_ece = expected_calibration_error(
            test_preds["y_true"], test_preds["y_prob"]
        )

        # fit temperature scaling on validation set
        temp_scaler = TemperatureScaling().to(self.device)
        val_logits = torch.tensor(val_preds["logits"], device=self.device)
        val_labels = torch.tensor(val_preds["y_true"], device=self.device).long()
        temperature = temp_scaler.fit(val_logits, val_labels)

        # apply to test set
        test_logits = torch.tensor(test_preds["logits"], device=self.device)
        cal_probs = temp_scaler.calibrate(test_logits)[:, 1].cpu().numpy()

        post_ece = expected_calibration_error(
            test_preds["y_true"], cal_probs
        )

        return {
            "pre_ece": pre_ece,
            "post_ece": post_ece,
            "temperature": temperature,
            "pre_brier": brier_score(test_preds["y_true"], test_preds["y_prob"]),
            "post_brier": brier_score(test_preds["y_true"], cal_probs),
            "reliability_pre": reliability_diagram_data(
                test_preds["y_true"], test_preds["y_prob"]
            ),
            "reliability_post": reliability_diagram_data(
                test_preds["y_true"], cal_probs
            ),
        }

    def save_predictions(
        self,
        dataloader: DataLoader,
        output_dir: str,
        split: str = "test",
    ):
        """Save per-sample predictions for reproducibility."""
        preds = self.predict(dataloader)
        os.makedirs(output_dir, exist_ok=True)

        np.savez(
            os.path.join(output_dir, f"{split}_seed{self.seed}.npz"),
            y_true=preds["y_true"],
            y_prob=preds["y_prob"],
            y_pred=preds["y_pred"],
            logits=preds["logits"],
        )
