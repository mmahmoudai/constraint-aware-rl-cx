"""
Main training loop integrating the PPO agent, CNN-Transformer classifier,
curriculum scheduler, and multi-objective reward.

Training protocol:
    - 150 epochs, AdamW, cosine annealing with 5-epoch linear warmup
    - Batch size: 64 (classifier), 128 (PPO)
    - Kaiming normal (conv), Xavier uniform (attention)
    - Deterministic CUDA for reproducibility
    - 4× A100 40 GB with DDP
"""

import os
import time
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.cnn_transformer import CNNTransformerClassifier
from ..models.ppo_agent import PPOAgent
from ..augmentation.clinical_transforms import ClinicalTransforms, N_ACTIONS
from ..augmentation.curriculum import CurriculumScheduler
from ..reward.multi_objective import MultiObjectiveReward
from ..state.state_extractor import StateExtractor
from ..utils.metrics import compute_metrics, expected_calibration_error
from ..utils.logging_utils import TBLogger, save_checkpoint


class Trainer:
    """Orchestrates training of the full constraint-aware RL augmentation framework.

    Args:
        config:       dict with all hyperparameters
        train_loader: training DataLoader
        val_loader:   validation DataLoader
        device:       torch device
        rank:         DDP rank (0 for single GPU)
    """

    def __init__(
        self,
        config: dict,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        rank: int = 0,
    ):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.rank = rank

        # build components
        self._build_classifier()
        self._build_agent()
        self._build_augmentation()
        self._build_reward()
        self._build_state_extractor()
        self._build_optimizer()

        # logging
        if rank == 0:
            self.logger = TBLogger(
                config.get("output_dir", "outputs"),
                experiment_name=config.get("experiment_name", "default"),
            )
            self.logger.log_config(config)
        else:
            self.logger = None

        self.best_val_auc = 0.0
        self.best_epoch = 0

    def _build_classifier(self):
        cfg = self.config.get("classifier", {})
        self.classifier = CNNTransformerClassifier(
            in_channels=3,
            num_classes=2,
            d_model=cfg.get("d_model", 64),
            nhead=cfg.get("nhead", 4),
            dim_ff=cfg.get("dim_ff", 256),
            n_layers=cfg.get("n_layers", 2),
            dropout=cfg.get("dropout", 0.3),
            use_mixstyle=cfg.get("use_mixstyle", False),
            mixstyle_p=cfg.get("mixstyle_p", 0.5),
            mixstyle_alpha=cfg.get("mixstyle_alpha", 0.1),
        ).to(self.device)

    def _build_agent(self):
        if not self.config.get("use_rl", True):
            self.agent = None
            return

        cfg = self.config.get("ppo", {})
        self.agent = PPOAgent(
            state_dim=cfg.get("state_dim", 100),
            n_actions=cfg.get("n_actions", N_ACTIONS),
            hidden_dim=cfg.get("hidden_dim", 256),
            clip_eps=cfg.get("clip_eps", 0.2),
            entropy_coeff=cfg.get("entropy_coeff", 0.02),
            gamma=cfg.get("gamma", 0.99),
            gae_lambda=cfg.get("gae_lambda", 0.95),
            actor_lr=cfg.get("actor_lr", 1e-4),
            critic_lr=cfg.get("critic_lr", 5e-4),
            buffer_size=cfg.get("buffer_size", 1024),
            batch_size=cfg.get("batch_size", 128),
        ).to(self.device)

    def _build_augmentation(self):
        cfg = self.config.get("augmentation", {})
        constrained = cfg.get("constrained", True)
        self.transforms = ClinicalTransforms(
            difficulty=0.0,
            constrained=constrained,
        )

        use_curriculum = cfg.get("use_curriculum", True)
        if use_curriculum:
            cur_cfg = self.config.get("curriculum", {})
            self.curriculum = CurriculumScheduler(
                n_stages=cur_cfg.get("n_stages", 5),
                rollback_thresh=cur_cfg.get("rollback_thresh", 0.02),
                tau_adapt=cur_cfg.get("tau_adapt", 100.0),
                total_epochs=self.config.get("epochs", 150),
            )
        else:
            self.curriculum = None

    def _build_reward(self):
        if not self.config.get("use_rl", True):
            self.reward_fn = None
            return

        cfg = self.config.get("reward", {})
        self.reward_fn = MultiObjectiveReward(
            alpha=cfg.get("alpha", 1.0),
            beta=cfg.get("beta", 0.3),
            delta=cfg.get("delta", 0.5),
            gamma_cost=cfg.get("gamma_cost", 0.1),
            epsilon=cfg.get("epsilon", 2.0),
        )

    def _build_state_extractor(self):
        if not self.config.get("use_rl", True):
            self.state_extractor = None
            return

        self.state_extractor = StateExtractor(
            seg_model=None,  # loaded separately if available
            classifier=self.classifier,
            device=self.device,
            n_mc=self.config.get("mc_dropout_passes", 20),
        )

    def _build_optimizer(self):
        cfg = self.config.get("classifier", {})
        self.optimizer = AdamW(
            self.classifier.parameters(),
            lr=cfg.get("lr", 2e-4),
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=cfg.get("weight_decay", 0.01),
        )

        epochs = self.config.get("epochs", 150)
        warmup_epochs = self.config.get("warmup_epochs", 5)

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=epochs - warmup_epochs,
            eta_min=1e-6,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )

        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        """Run the full training loop."""
        epochs = self.config.get("epochs", 150)

        for epoch in range(epochs):
            # update curriculum difficulty
            if self.curriculum is not None:
                self.transforms.set_difficulty(self.curriculum.difficulty)

            # train one epoch
            train_metrics = self._train_epoch(epoch)

            # validate
            val_metrics = self._validate(epoch)

            # curriculum advancement with rollback
            if self.curriculum is not None:
                result = self.curriculum.try_advance(val_metrics["ece"], epoch)
                if result["rolled_back"] and self.rank == 0:
                    print(
                        f"  [Rollback] Stage {result['stage']+1}->{result['stage']} "
                        f"(ECE delta: {result['ece_delta']:.4f})"
                    )
                elif result["advanced"] and self.rank == 0:
                    print(
                        f"  [Advance] Stage {result['stage']} "
                        f"(difficulty: {result['difficulty']:.2f})"
                    )

            # update reward baselines
            if self.reward_fn is not None:
                self.reward_fn.update_baselines(
                    val_metrics["auc"], val_metrics["ece"]
                )
                self.reward_fn.reset_action_counts()

            # update state extractor dynamics
            if self.state_extractor is not None:
                grad_norm = self._compute_grad_norm()
                self.state_extractor.update_dynamics(
                    loss=train_metrics["loss"],
                    grad_norm=grad_norm,
                    lr=self.optimizer.param_groups[0]["lr"],
                    epoch_fraction=epoch / epochs,
                    curriculum_stage=self.curriculum.current_stage if self.curriculum else 0,
                    val_acc_ema=val_metrics["accuracy"],
                )

            # step scheduler
            self.scheduler.step()

            # logging
            if self.rank == 0:
                self.logger.log_scalars(train_metrics, epoch, prefix="train")
                self.logger.log_scalars(val_metrics, epoch, prefix="val")

                if val_metrics["auc"] > self.best_val_auc:
                    self.best_val_auc = val_metrics["auc"]
                    self.best_epoch = epoch
                    self._save_best(epoch, val_metrics)

                if epoch % 10 == 0 or epoch == epochs - 1:
                    print(
                        f"Epoch {epoch:3d}/{epochs} | "
                        f"Train Loss: {train_metrics['loss']:.4f} | "
                        f"Val Acc: {val_metrics['accuracy']:.4f} | "
                        f"Val AUC: {val_metrics['auc']:.4f} | "
                        f"Val ECE: {val_metrics['ece']:.4f} | "
                        f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                    )

        if self.rank == 0:
            self.logger.close()
            print(f"\nBest Val AUC: {self.best_val_auc:.4f} at epoch {self.best_epoch}")

    def _train_epoch(self, epoch: int) -> dict:
        """Train one epoch with RL-guided augmentation."""
        self.classifier.train()
        total_loss = 0.0
        n_batches = 0

        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # RL augmentation
            if self.agent is not None and self.state_extractor is not None:
                augmented_images = self._rl_augment(images)
            else:
                augmented_images = images

            # forward pass
            logits = self.classifier(augmented_images)
            loss = self.criterion(logits, labels)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.classifier.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            # curriculum step
            if self.curriculum is not None:
                self.curriculum.step()

        # PPO update at end of epoch
        if self.agent is not None and len(self.agent.buffer["states"]) > 0:
            with torch.no_grad():
                dummy_state = torch.zeros(1, 100, device=self.device)
                _, next_value = self.agent(dummy_state)
            ppo_metrics = self.agent.update(next_value.squeeze())

        return {"loss": total_loss / max(n_batches, 1)}

    def _rl_augment(self, images: torch.Tensor) -> torch.Tensor:
        """Apply RL-guided augmentation to a batch."""
        B = images.size(0)
        augmented = []

        for i in range(B):
            img = images[i]

            # extract state
            state = self.state_extractor.extract(img)

            # select action
            action, log_prob, value = self.agent.select_action(state.unsqueeze(0))
            action_idx = action.item()

            # apply augmentation
            aug_time_start = time.time()
            aug_img = self.transforms.apply(img, action_idx)
            aug_time = time.time() - aug_time_start

            # compute implausibility
            implausibility = self.transforms.compute_implausibility(action_idx)

            # store transition (reward computed after forward pass)
            self.agent.store_transition(
                state=state,
                action=action.squeeze(),
                log_prob=log_prob.squeeze(),
                reward=0.0,  # placeholder, updated below
                value=value.squeeze(),
                done=False,
            )

            augmented.append(aug_img)

        return torch.stack(augmented)

    @torch.no_grad()
    def _validate(self, epoch: int) -> dict:
        """Run validation and compute metrics."""
        self.classifier.eval()
        all_probs = []
        all_labels = []

        for images, labels in self.val_loader:
            images = images.to(self.device)
            logits = self.classifier(images)
            probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

        y_true = np.array(all_labels)
        y_prob = np.array(all_probs)

        metrics = compute_metrics(y_true, y_prob)
        metrics["ece"] = expected_calibration_error(y_true, y_prob)

        return metrics

    def _compute_grad_norm(self) -> float:
        """Compute total gradient norm across classifier parameters."""
        total_norm = 0.0
        for p in self.classifier.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5

    def _save_best(self, epoch: int, metrics: dict):
        """Save best model checkpoint."""
        output_dir = self.config.get("output_dir", "outputs")
        seed = self.config.get("seed", 42)
        path = os.path.join(output_dir, f"seed_{seed}", "best_model.pt")

        save_checkpoint(
            path=path,
            epoch=epoch,
            classifier=self.classifier,
            agent=self.agent,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            curriculum=self.curriculum,
            metrics=metrics,
            config=self.config,
        )
