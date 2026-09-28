"""Configurable optimisation-based Neural Style Transfer engine."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Callable, List, Optional, Sequence

import torch

from .colour import luminance_only, match_luminance, recombine_luminance
from .losses import blended_gram_targets, content_loss, gram_matrix, style_loss, total_variation_loss
from .models import DEFAULT_CONTENT_LAYERS, DEFAULT_STYLE_LAYERS, VGGExtractor


@dataclass
class NSTConfig:
    arch: str = "vgg19"                 # vgg16 | vgg19
    pooling: str = "max"                # max | avg
    weights: str = "keras"              # keras | torchvision
    content_layers: List[str] = field(default_factory=lambda: list(DEFAULT_CONTENT_LAYERS))
    style_layers: List[str] = field(default_factory=lambda: list(DEFAULT_STYLE_LAYERS))
    style_layer_weights: Optional[List[float]] = None
    content_weight: float = 1.0         # alpha
    style_weight: float = 1e4           # beta
    tv_weight: float = 1e1              # gamma
    optimizer: str = "lbfgs"            # adam | lbfgs | sgd
    lr: float = 0.02                    # Adam / SGD learning rate (ignored by L-BFGS)
    steps: int = 300                    # number of loss evaluations (forward+backward passes)
    init: str = "content"               # content | noise | style
    preserve_colour: bool = False       # luminance-only transfer (Gatys et al., 2017)
    style_mix: Optional[List[float]] = None  # weights when several style images are given
    temporal_weight: float = 0.0        # for video: pull towards the previous stylised frame
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@lru_cache(maxsize=8)
def get_extractor(arch: str, pooling: str, weights: str, layers: tuple) -> VGGExtractor:
    return VGGExtractor(arch=arch, layers=layers, pooling=pooling, weights=weights)


@dataclass
class NSTResult:
    image: torch.Tensor
    log: List[dict]
    runtime_s: float
    config: NSTConfig


def stylise(
    content: torch.Tensor,
    styles: Sequence[torch.Tensor],
    cfg: NSTConfig,
    init_image: Optional[torch.Tensor] = None,
    temporal_target: Optional[torch.Tensor] = None,
    temporal_mask: Optional[torch.Tensor] = None,
    on_step: Optional[Callable[[int, dict], None]] = None,
    snapshot_at: Sequence[int] = (),
) -> NSTResult:
    """Run NST. ``content`` and each style are (1, 3, H, W) RGB tensors in [0, 1]."""
    torch.manual_seed(cfg.seed)
    t0 = time.perf_counter()

    layers = tuple(sorted(set(cfg.style_layers) | set(cfg.content_layers)))
    net = get_extractor(cfg.arch, cfg.pooling, cfg.weights, layers)

    content_in = luminance_only(content) if cfg.preserve_colour else content
    styles_in = ([match_luminance(luminance_only(s), content_in) for s in styles]
                 if cfg.preserve_colour else list(styles))

    with torch.no_grad():
        content_feats = net(content_in)
        style_feats = [net(s) for s in styles_in]
        if len(styles_in) == 1:
            grams = {l: gram_matrix(style_feats[0][l]) for l in cfg.style_layers}
        else:
            mix = cfg.style_mix or [1.0] * len(styles_in)
            grams = blended_gram_targets(style_feats, cfg.style_layers, mix)

    if init_image is not None:
        start = init_image.clone()
    elif cfg.init == "content":
        start = content_in.clone()
    elif cfg.init == "noise":
        start = torch.rand_like(content_in)
    elif cfg.init == "style":
        start = torch.nn.functional.interpolate(styles_in[0], size=content.shape[-2:], mode="bilinear",
                                                align_corners=False)
    else:
        raise ValueError(cfg.init)
    generated = start.clone().requires_grad_(True)

    if cfg.optimizer == "adam":
        opt = torch.optim.Adam([generated], lr=cfg.lr)
    elif cfg.optimizer == "sgd":
        opt = torch.optim.SGD([generated], lr=cfg.lr, momentum=0.9)
    elif cfg.optimizer == "lbfgs":
        # Full-batch quasi-Newton, as in Gatys et al. (2016). Run as one call
        # bounded by the evaluation budget; pixels are clamped only at the end.
        opt = torch.optim.LBFGS([generated], lr=1.0, max_iter=cfg.steps, max_eval=cfg.steps,
                                history_size=50, tolerance_grad=1e-9, tolerance_change=1e-12,
                                line_search_fn="strong_wolfe")
    else:
        raise ValueError(cfg.optimizer)

    log: List[dict] = []
    snapshots = {}
    evals = 0

    def closure():
        nonlocal evals
        opt.zero_grad()
        feats = net(generated)
        lc = content_loss(feats, content_feats, cfg.content_layers)
        ls = style_loss(feats, grams, cfg.style_layers, cfg.style_layer_weights)
        ltv = total_variation_loss(generated)
        total = cfg.content_weight * lc + cfg.style_weight * ls + cfg.tv_weight * ltv
        lt = torch.tensor(0.0)
        if temporal_target is not None and cfg.temporal_weight > 0:
            diff = (generated - temporal_target) ** 2
            if temporal_mask is not None:
                diff = diff * temporal_mask
            lt = diff.mean()
            total = total + cfg.temporal_weight * lt
        total.backward()
        evals += 1
        row = {
            "eval": evals,
            "time_s": time.perf_counter() - t0,
            "total": total.item(),
            "content": lc.item(),
            "style": ls.item(),
            "tv": ltv.item(),
            "temporal": float(lt.detach()),
            "w_content": cfg.content_weight * lc.item(),
            "w_style": cfg.style_weight * ls.item(),
        }
        log.append(row)
        if on_step:
            on_step(evals, row)
        if evals in snapshot_at:
            snapshots[evals] = generated.detach().clamp(0, 1).clone()
        return total

    if cfg.optimizer == "lbfgs":
        opt.step(closure)
    else:
        while evals < cfg.steps:
            opt.step(closure)
            with torch.no_grad():
                generated.clamp_(0, 1)

    out = generated.detach().clamp(0, 1)
    if cfg.preserve_colour:
        out = recombine_luminance(out, content)
    result = NSTResult(image=out, log=log, runtime_s=time.perf_counter() - t0, config=cfg)
    result.snapshots = snapshots  # type: ignore[attr-defined]
    return result
