"""Quantitative evaluation metrics for stylised images.

Deep-feature losses alone are circular (the optimiser minimises them), so the
evaluation combines one fixed deep metric with network-independent ones:

* ``ssim``          - structural similarity to the content image (Wang et al., 2004), greyscale.
* ``edge_f1``       - F1 overlap of Canny edges (content vs output, 1-px tolerance).
* ``colour_w1``     - mean 1-D Wasserstein distance between output and style in CIELAB.
* ``texture_chi2``  - chi-squared distance between LBP texture histograms of output and style.
* ``eval_style``    - log10 Gram distance measured by a *fixed* evaluator network
                      (VGG-19, avg-pool, normalised), identical for every run.
* ``tv``            - total variation of the output (high-frequency noise / artefacts).
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import binary_dilation
from scipy.stats import wasserstein_distance
from skimage.color import rgb2gray, rgb2lab
from skimage.feature import canny, local_binary_pattern
from skimage.metrics import structural_similarity
from skimage.transform import resize

from .losses import gram_matrix
from .models import VGGExtractor

_EVAL_LAYERS = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]
_evaluator = None


def _np(img: torch.Tensor) -> np.ndarray:
    return img.detach().cpu().squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy().astype(np.float64)


def ssim(content: np.ndarray, output: np.ndarray) -> float:
    return float(structural_similarity(rgb2gray(content), rgb2gray(output), data_range=1.0))


def edge_f1(content: np.ndarray, output: np.ndarray, sigma: float = 2.0) -> float:
    e1, e2 = canny(rgb2gray(content), sigma=sigma), canny(rgb2gray(output), sigma=sigma)
    if e1.sum() == 0 or e2.sum() == 0:
        return 0.0
    precision = (e2 & binary_dilation(e1)).sum() / e2.sum()
    recall = (e1 & binary_dilation(e2)).sum() / e1.sum()
    return float(2 * precision * recall / (precision + recall + 1e-12))


def colour_w1(style: np.ndarray, output: np.ndarray) -> float:
    ls, lo = rgb2lab(style).reshape(-1, 3), rgb2lab(output).reshape(-1, 3)
    rng = np.random.default_rng(0)
    ls = ls[rng.choice(len(ls), min(len(ls), 40000), replace=False)]
    lo = lo[rng.choice(len(lo), min(len(lo), 40000), replace=False)]
    return float(np.mean([wasserstein_distance(ls[:, i], lo[:, i]) for i in range(3)]))


def _lbp_hist(img: np.ndarray) -> np.ndarray:
    g = (rgb2gray(img) * 255).astype(np.uint8)
    hists = []
    for p, r in [(8, 1), (16, 2)]:
        lbp = local_binary_pattern(g, p, r, method="uniform")
        h, _ = np.histogram(lbp, bins=p + 2, range=(0, p + 2), density=True)
        hists.append(h)
    return np.concatenate(hists)


def texture_chi2(style: np.ndarray, output: np.ndarray) -> float:
    a, b = _lbp_hist(style), _lbp_hist(output)
    return float(0.5 * np.sum((a - b) ** 2 / (a + b + 1e-10)))


def eval_style(style: torch.Tensor, output: torch.Tensor) -> float:
    global _evaluator
    if _evaluator is None:
        _evaluator = VGGExtractor("vgg19", _EVAL_LAYERS, pooling="avg", weights="keras")
    with torch.no_grad():
        fs, fo = _evaluator(style), _evaluator(output)
        d = sum(torch.mean((gram_matrix(fs[l]) - gram_matrix(fo[l])) ** 2).item() for l in _EVAL_LAYERS)
    return float(np.log10(d + 1e-12))


def total_variation(output: np.ndarray) -> float:
    return float(np.abs(np.diff(output, axis=0)).mean() + np.abs(np.diff(output, axis=1)).mean())


def evaluate(content: torch.Tensor, style: torch.Tensor, output: torch.Tensor) -> dict:
    c, s, o = _np(content), _np(style), _np(output)
    s_resized = resize(s, o.shape[:2], anti_aliasing=True)
    return {
        "ssim": ssim(c, o),
        "edge_f1": edge_f1(c, o),
        "colour_w1": colour_w1(s, o),
        "texture_chi2": texture_chi2(s_resized, o),
        "eval_style": eval_style(style, output),
        "tv": total_variation(o),
    }
