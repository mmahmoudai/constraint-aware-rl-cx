"""
MixStyle: domain generalization via probabilistic feature-level style mixing.

Reference: Zhou et al., "Domain Generalization with MixStyle", ICLR 2021.

Applied after the first and second convolutional blocks of the CNN backbone
with mixing probability p = 0.5 and default hyperparameters.
"""

import torch
import torch.nn as nn


class MixStyle(nn.Module):
    """Probabilistically mix feature statistics across instances within a mini-batch.

    For each feature map in the batch, with probability p, replace its instance-level
    mean and std with a convex combination of its own statistics and those of a
    randomly shuffled instance from the same batch.

    Args:
        p:     probability of applying MixStyle to each sample (default 0.5)
        alpha: Beta distribution parameter for mixing coefficient (default 0.1)
        eps:   small constant for numerical stability
    """

    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x

        B = x.size(0)
        if B <= 1:
            return x

        # decide which samples to mix (Bernoulli mask)
        mask = torch.rand(B, device=x.device) < self.p
        if not mask.any():
            return x

        # compute instance-level statistics (over spatial dims)
        mu = x.mean(dim=[2, 3], keepdim=True)
        sig = (x.var(dim=[2, 3], keepdim=True) + self.eps).sqrt()

        # normalize
        x_normed = (x - mu) / sig

        # sample mixing coefficient from Beta(alpha, alpha)
        lmda = torch.distributions.Beta(self.alpha, self.alpha).sample(
            (B, 1, 1, 1)
        ).to(x.device)

        # random permutation for pairing
        perm = torch.randperm(B, device=x.device)
        mu_mix = lmda * mu + (1 - lmda) * mu[perm]
        sig_mix = lmda * sig + (1 - lmda) * sig[perm]

        # apply mixed statistics only to selected samples
        x_mixed = x_normed * sig_mix + mu_mix

        # blend: apply MixStyle only where mask is True
        mask = mask.view(B, 1, 1, 1).float()
        return mask * x_mixed + (1 - mask) * x
