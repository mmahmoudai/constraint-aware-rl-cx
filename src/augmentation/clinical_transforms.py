"""
Radiologically bounded augmentation transforms for chest X-rays.

Action space: 5 transform types × 12 intensity levels = 60 discrete actions.

Transform types and clinical bounds:
    0. Rotation:       θ ∈ [-8°, 8°]     (patient positioning variability)
    1. Brightness:     β ∈ [0.9, 1.2]    (exposure variation)
    2. Contrast:       γ ∈ [0.8, 1.2]    (opacity visibility)
    3. Gaussian Noise: σ ∈ [0, 0.03]     (acquisition noise)
    4. Gaussian Blur:  σ_b ∈ [0, 1.0]    (minor motion artifacts)

Horizontal flipping is deliberately excluded to preserve laterality.
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


TRANSFORM_NAMES = ["rotation", "brightness", "contrast", "noise", "blur"]
N_TRANSFORMS = 5
N_INTENSITY_LEVELS = 12
N_ACTIONS = N_TRANSFORMS * N_INTENSITY_LEVELS  # 60

# Clinical parameter bounds: (min_value, max_value) for each transform
CLINICAL_BOUNDS = {
    "rotation": (-8.0, 8.0),        # degrees
    "brightness": (0.9, 1.2),       # multiplicative factor
    "contrast": (0.8, 1.2),         # multiplicative factor
    "noise": (0.0, 0.03),           # Gaussian σ
    "blur": (0.0, 1.0),             # Gaussian kernel σ
}


def intensity_to_param(transform_type: int, intensity_level: int) -> float:
    """Map a discrete intensity level (0-11) to a continuous parameter value
    within the clinical bounds for the given transform type.

    Level 0 = minimum (identity-like), level 11 = maximum clinical bound.
    """
    name = TRANSFORM_NAMES[transform_type]
    lo, hi = CLINICAL_BOUNDS[name]
    fraction = intensity_level / (N_INTENSITY_LEVELS - 1)
    return lo + fraction * (hi - lo)


def apply_rotation(image: torch.Tensor, angle_deg: float) -> torch.Tensor:
    """Rotate image by angle_deg degrees with bilinear interpolation."""
    return TF.rotate(image, angle_deg, interpolation=TF.InterpolationMode.BILINEAR)


def apply_brightness(image: torch.Tensor, factor: float) -> torch.Tensor:
    """Adjust brightness by multiplicative factor."""
    return TF.adjust_brightness(image, factor)


def apply_contrast(image: torch.Tensor, factor: float) -> torch.Tensor:
    """Adjust contrast by multiplicative factor."""
    return TF.adjust_contrast(image, factor)


def apply_noise(image: torch.Tensor, sigma: float) -> torch.Tensor:
    """Add Gaussian noise with standard deviation sigma."""
    if sigma <= 0:
        return image
    noise = torch.randn_like(image) * sigma
    return torch.clamp(image + noise, 0.0, 1.0)


def apply_blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply Gaussian blur with kernel standard deviation sigma."""
    if sigma <= 0:
        return image
    kernel_size = int(math.ceil(sigma * 6)) | 1  # ensure odd
    kernel_size = max(kernel_size, 3)
    return TF.gaussian_blur(image, kernel_size=[kernel_size, kernel_size], sigma=[sigma])


TRANSFORM_FNS = [apply_rotation, apply_brightness, apply_contrast, apply_noise, apply_blur]


class ClinicalTransforms:
    """Applies a single clinically-bounded augmentation given a flat action index.

    Usage:
        transforms = ClinicalTransforms()
        augmented = transforms.apply(image, action_idx=23)

    or with curriculum difficulty scaling:
        transforms = ClinicalTransforms(difficulty=0.5)
        augmented = transforms.apply(image, action_idx=23)
    """

    def __init__(
        self,
        difficulty: float = 1.0,
        constrained: bool = True,
    ):
        """
        Args:
            difficulty: curriculum difficulty ∈ [0, 1]. Scales intensity levels.
            constrained: if False, allow parameters beyond clinical bounds
                         (used for the unconstrained RL baseline).
        """
        self.difficulty = max(0.0, min(1.0, difficulty))
        self.constrained = constrained

        if not constrained:
            # Unconstrained baseline: widen bounds by 3×
            self.bounds = {
                "rotation": (-24.0, 24.0),
                "brightness": (0.5, 1.8),
                "contrast": (0.4, 2.0),
                "noise": (0.0, 0.10),
                "blur": (0.0, 3.0),
            }
        else:
            self.bounds = CLINICAL_BOUNDS.copy()

    def _get_param(self, transform_type: int, intensity_level: int) -> float:
        """Compute the actual parameter value, scaled by difficulty."""
        name = TRANSFORM_NAMES[transform_type]
        lo, hi = self.bounds[name]

        # scale intensity by curriculum difficulty
        effective_level = intensity_level * self.difficulty
        fraction = effective_level / (N_INTENSITY_LEVELS - 1)
        return lo + fraction * (hi - lo)

    def apply(self, image: torch.Tensor, action_idx: int) -> torch.Tensor:
        """Apply the augmentation corresponding to action_idx.

        Args:
            image: (C, H, W) tensor in [0, 1]
            action_idx: integer in [0, 59]

        Returns:
            augmented image tensor
        """
        transform_type = action_idx // N_INTENSITY_LEVELS
        intensity_level = action_idx % N_INTENSITY_LEVELS
        param = self._get_param(transform_type, intensity_level)

        if transform_type == 0:
            return apply_rotation(image, param)
        elif transform_type == 1:
            return apply_brightness(image, param)
        elif transform_type == 2:
            return apply_contrast(image, param)
        elif transform_type == 3:
            return apply_noise(image, param)
        elif transform_type == 4:
            return apply_blur(image, param)
        else:
            return image

    def apply_batch(
        self, images: torch.Tensor, action_indices: torch.Tensor
    ) -> torch.Tensor:
        """Apply per-image augmentations to a batch.

        Args:
            images: (B, C, H, W) tensor
            action_indices: (B,) integer tensor

        Returns:
            (B, C, H, W) augmented batch
        """
        augmented = []
        for i in range(images.size(0)):
            aug = self.apply(images[i], action_indices[i].item())
            augmented.append(aug)
        return torch.stack(augmented)

    def set_difficulty(self, difficulty: float):
        self.difficulty = max(0.0, min(1.0, difficulty))

    def compute_implausibility(self, action_idx: int) -> float:
        """Compute clinical implausibility penalty for an action.

        Returns 0 if within clinical bounds, positive penalty otherwise.
        Only relevant for unconstrained baseline.
        """
        if self.constrained:
            return 0.0

        transform_type = action_idx // N_INTENSITY_LEVELS
        intensity_level = action_idx % N_INTENSITY_LEVELS
        name = TRANSFORM_NAMES[transform_type]
        param = self._get_param(transform_type, intensity_level)

        clin_lo, clin_hi = CLINICAL_BOUNDS[name]
        if param < clin_lo:
            return abs(param - clin_lo)
        elif param > clin_hi:
            return abs(param - clin_hi)
        return 0.0


class RandomAugmentation:
    """Baseline: uniformly random augmentation within clinical bounds."""

    def __init__(self, difficulty: float = 1.0):
        self.transforms = ClinicalTransforms(difficulty=difficulty)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        action_idx = torch.randint(0, N_ACTIONS, (1,)).item()
        return self.transforms.apply(image, action_idx)


class RandAugmentCXR:
    """RandAugment adapted for CXR: apply N random transforms at magnitude M.

    N: number of transforms per image
    M: magnitude level (0-12), mapped to intensity
    """

    def __init__(self, n: int = 2, m: int = 9):
        self.n = n
        self.m = min(m, N_INTENSITY_LEVELS - 1)
        self.transforms = ClinicalTransforms(difficulty=1.0)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        for _ in range(self.n):
            transform_type = torch.randint(0, N_TRANSFORMS, (1,)).item()
            action_idx = transform_type * N_INTENSITY_LEVELS + self.m
            image = self.transforms.apply(image, action_idx)
        return image


class TrivialAugmentCXR:
    """TrivialAugment: single random transform at random intensity per image."""

    def __init__(self):
        self.transforms = ClinicalTransforms(difficulty=1.0)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        action_idx = torch.randint(0, N_ACTIONS, (1,)).item()
        return self.transforms.apply(image, action_idx)
