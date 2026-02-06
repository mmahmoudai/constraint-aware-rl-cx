"""
CNN-Transformer hybrid classifier for chest X-ray pneumonia detection.

Architecture:
    CNN backbone: Conv blocks 16 -> 32 -> 64 channels
    Patch tokenization: 784 patches of dimension 64
    Transformer encoder: 2 layers, 4 heads, key/query dim 32, FFN hidden 256
    Total parameters: 340,899
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBackbone(nn.Module):
    """Three-block convolutional feature extractor.

    Each block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> MaxPool
    Input:  (B, 3, 224, 224)
    Output: (B, 64, 28, 28)
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

    def forward_with_intermediates(self, x: torch.Tensor):
        """Return feature maps after each block (used by MixStyle)."""
        f1 = self.block1(x)
        f2 = self.block2(f1)
        f3 = self.block3(f2)
        return f1, f2, f3


class PatchTokenizer(nn.Module):
    """Reshape CNN feature maps into a sequence of patch tokens.

    Input:  (B, 64, 28, 28)
    Output: (B, 784, 64)  -- 784 = 28*28 patches, each of dim 64
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # (B, C, H, W) -> (B, H*W, C)
        return x.flatten(2).transpose(1, 2)


class TransformerEncoder(nn.Module):
    """Transformer encoder with learnable positional embeddings.

    Uses internal attention dimension of nhead × head_dim = 4 × 32 = 128,
    with input/output projection from/to d_model = 64.

    Parameters:
        d_model:  64  (patch / token dimension)
        nhead:    4   (attention heads)
        head_dim: 32  (key/query dimension per head)
        dim_ff:   256 (FFN hidden size)
        n_layers: 2
        dropout:  0.3
    """

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        head_dim: int = 32,
        dim_ff: int = 256,
        n_layers: int = 2,
        dropout: float = 0.3,
        max_seq_len: int = 784,
    ):
        super().__init__()
        attn_dim = nhead * head_dim  # 128
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, nhead, head_dim, attn_dim, dim_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_embed[:, : x.size(1), :]
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer block with cross-dimension attention projection."""

    def __init__(self, d_model, nhead, head_dim, attn_dim, dim_ff, dropout):
        super().__init__()
        self.nhead = nhead
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, attn_dim)
        self.k_proj = nn.Linear(d_model, attn_dim)
        self.v_proj = nn.Linear(d_model, attn_dim)
        self.out_proj = nn.Linear(attn_dim, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        # self-attention with residual
        h = self.norm1(x)
        q = self.q_proj(h).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, T, self.nhead, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, -1)
        x = x + self.out_proj(out)

        # FFN with residual
        x = x + self.ffn(self.norm2(x))
        return x


class GlobalAveragePooling(nn.Module):
    """Pool sequence tokens into a single vector: (B, T, D) -> (B, D)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)


class ClassificationHead(nn.Module):
    """Two-layer MLP: Dense(64) -> ReLU -> Dropout -> Dense(128) -> ReLU -> Dropout -> Dense(2)."""

    def __init__(self, d_model: int = 64, hidden: int = 128, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNNTransformerClassifier(nn.Module):
    """Full CNN-Transformer classifier (340,899 parameters).

    Forward pass:
        I_norm (B, 3, 224, 224)
        -> CNN backbone -> (B, 64, 28, 28)
        -> Patch tokenization -> (B, 784, 64)
        -> Transformer encoder (2 layers, 4 heads) -> (B, 784, 64)
        -> Global average pooling -> (B, 64)
        -> Classification head -> (B, 2)
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        d_model: int = 64,
        nhead: int = 4,
        dim_ff: int = 256,
        n_layers: int = 2,
        dropout: float = 0.3,
        use_mixstyle: bool = False,
        mixstyle_p: float = 0.5,
        mixstyle_alpha: float = 0.1,
    ):
        super().__init__()
        self.cnn = CNNBackbone(in_channels)
        self.tokenizer = PatchTokenizer()
        self.transformer = TransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            dim_ff=dim_ff,
            n_layers=n_layers,
            dropout=dropout,
        )
        self.pool = GlobalAveragePooling()
        self.head = ClassificationHead(d_model=d_model, num_classes=num_classes, dropout=dropout)

        self.use_mixstyle = use_mixstyle
        if use_mixstyle:
            from .mixstyle import MixStyle

            self.mixstyle = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_mixstyle and self.training:
            f1, f2, f3 = self.cnn.forward_with_intermediates(x)
            f1 = self.mixstyle(f1)
            f2 = self.mixstyle(f2)
            features = f3
        else:
            features = self.cnn(x)

        tokens = self.tokenizer(features)
        encoded = self.transformer(tokens)
        pooled = self.pool(encoded)
        logits = self.head(pooled)
        return logits

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return pooled features before classification head."""
        features = self.cnn(x)
        tokens = self.tokenizer(features)
        encoded = self.transformer(tokens)
        pooled = self.pool(encoded)
        return pooled

    def mc_dropout_predict(
        self, x: torch.Tensor, n_forward: int = 20
    ) -> torch.Tensor:
        """MC Dropout prediction: average softmax over n_forward stochastic passes.

        Returns:
            (B, num_classes) averaged probability estimates.
        """
        self.train()  # enable dropout
        preds = []
        with torch.no_grad():
            for _ in range(n_forward):
                logits = self.forward(x)
                preds.append(F.softmax(logits, dim=-1))
        self.eval()
        return torch.stack(preds).mean(dim=0)

    def mc_dropout_uncertainty(
        self, x: torch.Tensor, n_forward: int = 20
    ) -> dict:
        """Compute MC Dropout uncertainty statistics.

        Returns dict with keys:
            mean_probs: (B, C) mean predicted probabilities
            variance:   (B, C) variance across forward passes
            entropy:    (B,) predictive entropy
            confidence_hist: (B, 8) histogram of max-class confidence across passes
        """
        self.train()
        all_preds = []
        with torch.no_grad():
            for _ in range(n_forward):
                logits = self.forward(x)
                all_preds.append(F.softmax(logits, dim=-1))
        self.eval()

        stacked = torch.stack(all_preds)  # (T, B, C)
        mean_probs = stacked.mean(dim=0)
        variance = stacked.var(dim=0)

        # predictive entropy
        entropy = -(mean_probs * (mean_probs + 1e-10).log()).sum(dim=-1)

        # confidence histograms: bin max-class prob across T passes into 8 bins
        max_confs = stacked.max(dim=-1).values  # (T, B)
        max_confs = max_confs.transpose(0, 1)  # (B, T)
        bins = torch.linspace(0, 1, 9, device=x.device)
        hist = torch.zeros(x.size(0), 8, device=x.device)
        for i in range(8):
            mask = (max_confs >= bins[i]) & (max_confs < bins[i + 1])
            hist[:, i] = mask.float().sum(dim=-1)
        hist = hist / n_forward

        return {
            "mean_probs": mean_probs,
            "variance": variance,
            "entropy": entropy,
            "confidence_hist": hist,
        }
