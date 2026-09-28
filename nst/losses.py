"""Loss functions for optimisation-based NST (Gatys et al., 2016)."""
from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn.functional as F


def gram_matrix(feature_map: torch.Tensor) -> torch.Tensor:
    """Normalised Gram matrix G = F F^T / (C*H*W) of a (B, C, H, W) feature map."""
    b, c, h, w = feature_map.shape
    f = feature_map.reshape(b, c, h * w)
    return torch.bmm(f, f.transpose(1, 2)) / (c * h * w)


def content_loss(gen: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor],
                 layers: Sequence[str]) -> torch.Tensor:
    return sum(F.mse_loss(gen[l], target[l]) for l in layers) / len(layers)


def style_loss(gen: Dict[str, torch.Tensor], target_grams: Dict[str, torch.Tensor],
               layers: Sequence[str], layer_weights: Sequence[float] | None = None) -> torch.Tensor:
    if layer_weights is None:
        layer_weights = [1.0] * len(layers)
    total = 0.0
    for l, w in zip(layers, layer_weights):
        total = total + w * F.mse_loss(gram_matrix(gen[l]), target_grams[l])
    return total


def total_variation_loss(img: torch.Tensor) -> torch.Tensor:
    """Anisotropic TV: mean absolute difference between neighbouring pixels."""
    dv = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
    dh = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
    return dv + dh


def blended_gram_targets(style_feature_list, layers, mix):
    """Multi-style targets: a convex combination of each style's Gram matrices."""
    mix = torch.tensor(mix, dtype=torch.float32)
    mix = mix / mix.sum()
    return {
        l: sum(float(m) * gram_matrix(feats[l]) for m, feats in zip(mix, style_feature_list))
        for l in layers
    }
