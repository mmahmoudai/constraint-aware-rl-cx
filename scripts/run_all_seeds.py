"""
Run training across all 5 seeds sequentially.

Usage:
    python scripts/run_all_seeds.py --config configs/default.yaml
"""

import argparse
import subprocess
import sys

SEEDS = [42, 123, 2025, 314, 555]


def main():
    parser = argparse.ArgumentParser(description="Run all seeds")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--distributed", action="store_true",
                        help="Use torchrun for multi-GPU")
    parser.add_argument("--nproc", type=int, default=4,
                        help="Number of GPUs for distributed training")
    args = parser.parse_args()

    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"Training with seed {seed}")
        print(f"{'='*60}\n")

        if args.distributed:
            cmd = [
                sys.executable, "-m", "torch.distributed.run",
                f"--nproc_per_node={args.nproc}",
                "scripts/train.py",
                "--config", args.config,
                "--seed", str(seed),
            ]
        else:
            cmd = [
                sys.executable, "scripts/train.py",
                "--config", args.config,
                "--seed", str(seed),
            ]

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Training failed for seed {seed} (exit code {result.returncode})")
            continue

    print(f"\nAll seeds completed.")


if __name__ == "__main__":
    main()
