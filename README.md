# Adaptive Augmentation via Constrained RL for Cross-Site Chest X-Ray Pneumonia Classification

This repository contains the code, configuration files, and evaluation scripts for the constraint-aware reinforcement learning framework described in the accompanying paper.

## Overview

A PPO agent learns to select radiologically bounded augmentations for chest X-ray pneumonia classification. The agent operates over 60 discrete actions (5 transform types × 12 intensity levels) and receives a multi-objective reward balancing accuracy, calibration (ECE), augmentation diversity, and computational cost. A five-stage curriculum with safety rollback governs augmentation intensity.

## Repository Structure

```
constraint-aware-rl-cxr/
├── configs/
│   ├── default.yaml                 # Full framework config
│   └── baselines/
│       ├── no_aug.yaml
│       ├── random_aug.yaml
│       ├── autoaugment.yaml
│       ├── randaugment.yaml
│       ├── trivialaugment.yaml
│       ├── randaug_curriculum.yaml
│       ├── rl_unconstrained.yaml
│       └── mixstyle_randaug.yaml
├── src/
│   ├── models/
│   │   ├── cnn_transformer.py       # CNN-Transformer classifier (340,899 params)
│   │   ├── ppo_agent.py             # PPO-based RL agent (173,117 params)
│   │   └── mixstyle.py              # MixStyle domain generalization module
│   ├── augmentation/
│   │   ├── clinical_transforms.py   # Radiologically bounded transforms
│   │   └── curriculum.py            # Curriculum scheduler with rollback
│   ├── reward/
│   │   └── multi_objective.py       # Multi-objective reward computation
│   ├── state/
│   │   └── state_extractor.py       # 100-dim modality-aware state vector
│   ├── data/
│   │   ├── datasets.py              # CheXpert, MIMIC-CXR, external loaders
│   │   └── preprocessing.py         # Normalization, quality filtering
│   ├── training/
│   │   ├── trainer.py               # Main training loop with RL + classifier
│   │   └── evaluator.py             # Evaluation, calibration, statistical tests
│   └── utils/
│       ├── metrics.py               # AUC, ECE, Brier, McNemar, DeLong
│       ├── calibration.py           # Temperature scaling, reliability diagrams
│       └── logging_utils.py         # TensorBoard + CSV logging
├── scripts/
│   ├── train.py                     # Entry point for training
│   ├── evaluate.py                  # Entry point for evaluation
│   ├── inject_label_noise.py        # Label-noise stress test script
│   └── run_all_seeds.py             # Run all 5 seeds sequentially
├── requirements.txt
└── README.md
```

## Setup

### Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA 12.1
- 4× NVIDIA A100 40 GB (for full reproduction; single GPU works with reduced batch size)

```bash
pip install -r requirements.txt
```

### Data Preparation

1. **CheXpert**: Download from https://stanfordmlgroup.github.io/competitions/chexpert/
2. **MIMIC-CXR**: Access via PhysioNet (requires credentialed access): https://physionet.org/content/mimic-cxr-jpg/
3. **ChestX-ray14**: Download from https://nihcc.app.box.com/v/ChestXray-NIHCC
4. **VinDr-CXR**: Download from https://physionet.org/content/vindr-cxr/
5. **Kermany Pediatric**: Download from https://data.mendeley.com/datasets/rscbjbr9sj/2

Place datasets under a common root and update `data_root` in config files:

```
/data/
├── chexpert/
├── mimic-cxr-jpg/
├── chestxray14/
├── vindr-cxr/
└── kermany/
```

## Training

### Full framework (default config)

```bash
# Single GPU
python scripts/train.py --config configs/default.yaml --seed 42

# Multi-GPU (4× A100)
torchrun --nproc_per_node=4 scripts/train.py --config configs/default.yaml --seed 42
```

### Run all seeds

```bash
python scripts/run_all_seeds.py --config configs/default.yaml
```

### Baselines

```bash
python scripts/train.py --config configs/baselines/no_aug.yaml --seed 42
python scripts/train.py --config configs/baselines/randaugment.yaml --seed 42
python scripts/train.py --config configs/baselines/randaug_curriculum.yaml --seed 42
python scripts/train.py --config configs/baselines/rl_unconstrained.yaml --seed 42
python scripts/train.py --config configs/baselines/mixstyle_randaug.yaml --seed 42
# ... etc.
```

## Evaluation

```bash
# Internal test set
python scripts/evaluate.py --config configs/default.yaml --checkpoint outputs/seed_42/best_model.pt --split test

# External validation
python scripts/evaluate.py --config configs/default.yaml --checkpoint outputs/seed_42/best_model.pt --external chestxray14
python scripts/evaluate.py --config configs/default.yaml --checkpoint outputs/seed_42/best_model.pt --external vindrcxr
python scripts/evaluate.py --config configs/default.yaml --checkpoint outputs/seed_42/best_model.pt --external kermany
```

## Label-Noise Stress Test

```bash
python scripts/inject_label_noise.py --config configs/default.yaml --noise_rates 0.0 0.05 0.10 0.15 0.20 --seeds 42 123 2025
```

## Key Hyperparameters

| Component | Parameter | Value |
|-----------|-----------|-------|
| PPO Agent | Actor LR | 1e-4 |
| PPO Agent | Critic LR | 5e-4 |
| PPO Agent | Batch / Buffer | 128 / 1024 |
| PPO Agent | Clip ε / Entropy | 0.2 / 0.02 |
| Classifier | Optimizer / LR | AdamW / 2e-4 |
| Classifier | Epochs / Batch | 150 / 64 |
| Classifier | Dropout / WD | 0.3 / 0.01 |
| Reward | α, β, δ, γ_c, ε | 1.0, 0.3, 0.5, 0.1, 2.0 |
| Curriculum | Stages / Rollback τ | 5 / 0.02 |

## Reproducing Paper Results

To reproduce all experiments from scratch:

```bash
# 1. Train full framework across 5 seeds
python scripts/run_all_seeds.py --config configs/default.yaml

# 2. Train all baselines across 5 seeds
for baseline in no_aug random_aug autoaugment randaugment trivialaugment randaug_curriculum rl_unconstrained mixstyle_randaug; do
    python scripts/run_all_seeds.py --config configs/baselines/${baseline}.yaml
done

# 3. Run label-noise experiments
python scripts/inject_label_noise.py --config configs/default.yaml --noise_rates 0.0 0.05 0.10 0.15 0.20

# 4. Evaluate all checkpoints
python scripts/evaluate.py --all_checkpoints --output results/
```

## License

This project is released under the MIT License.
