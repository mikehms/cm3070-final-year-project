"""Frozen VGG feature extractors for Neural Style Transfer.

Supports VGG-16 and VGG-19, max or average pooling, and two sources of
ImageNet-pretrained weights:

* ``keras``       - the original Keras/Caffe VGG weights (``*_notop.h5``) that
                    Chollet (2018) uses. These expect BGR input in the 0-255
                    range with the ImageNet channel mean subtracted.
* ``torchvision`` - torchvision's ``VGG16_Weights`` / ``VGG19_Weights``. These
                    expect RGB input in 0-1, normalised by mean/std.

Whichever source is used, the extractor takes an RGB tensor in [0, 1] and does
its own preprocessing, so the generated image can always be optimised in the
same pixel space.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import torch
from torch import nn

# Block layout of each architecture: number of conv layers per block.
VGG_BLOCKS = {
    "vgg16": [2, 2, 3, 3, 3],
    "vgg19": [2, 2, 4, 4, 4],
}
BLOCK_CHANNELS = [64, 128, 256, 512, 512]

DEFAULT_STYLE_LAYERS = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]
DEFAULT_CONTENT_LAYERS = ["conv4_2"]

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"


def build_vgg_features(arch: str, pooling: str = "max") -> tuple[nn.Sequential, List[str]]:
    """Create the convolutional part of VGG and a parallel list of layer names."""
    if arch not in VGG_BLOCKS:
        raise ValueError(f"Unknown architecture {arch!r}; choose from {list(VGG_BLOCKS)}")
    layers: List[nn.Module] = []
    names: List[str] = []
    in_ch = 3
    for b, (n_conv, out_ch) in enumerate(zip(VGG_BLOCKS[arch], BLOCK_CHANNELS), start=1):
        for c in range(1, n_conv + 1):
            layers += [nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(inplace=False)]
            # Caffe applies ReLU in place, so "convX_Y" in Gatys et al. means the
            # *rectified* output. We follow that convention.
            names += [f"conv{b}_{c}_linear", f"conv{b}_{c}"]
            in_ch = out_ch
        pool = nn.MaxPool2d(2, 2) if pooling == "max" else nn.AvgPool2d(2, 2)
        layers.append(pool)
        names.append(f"pool{b}")
    return nn.Sequential(*layers), names


def _load_keras_weights(features: nn.Sequential, names: List[str], arch: str) -> None:
    import h5py

    path = WEIGHTS_DIR / f"{arch}_notop.h5"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download it from https://github.com/fchollet/"
            f"deep-learning-models/releases/download/v0.1/{arch}_weights_tf_dim_ordering_tf_kernels_notop.h5"
        )
    with h5py.File(path, "r") as f:
        for module, name in zip(features, names):
            if not isinstance(module, nn.Conv2d):
                continue
            if not name.endswith("_linear"):
                continue
            b, c = name[4:].split("_")[:2]
            key = f"block{b}_conv{c}"
            group = f[key]
            w = [group[k][()] for k in group.keys() if "_W" in k][0]  # (H, W, in, out)
            bias = [group[k][()] for k in group.keys() if "_b" in k][0]
            module.weight.data = torch.from_numpy(w).permute(3, 2, 0, 1).contiguous().float()
            module.bias.data = torch.from_numpy(bias).float()


def _load_torchvision_weights(features: nn.Sequential, arch: str) -> None:
    from torchvision import models

    ctor = getattr(models, arch)
    weights = getattr(models, f"{arch.upper()}_Weights").DEFAULT
    reference = ctor(weights=weights).features
    ref_convs = [m for m in reference if isinstance(m, nn.Conv2d)]
    my_convs = [m for m in features if isinstance(m, nn.Conv2d)]
    for mine, ref in zip(my_convs, ref_convs):
        mine.weight.data.copy_(ref.weight.data)
        mine.bias.data.copy_(ref.bias.data)


class VGGExtractor(nn.Module):
    """Frozen VGG network that returns activations for the requested layers."""

    def __init__(
        self,
        arch: str = "vgg19",
        layers: Iterable[str] = DEFAULT_STYLE_LAYERS + DEFAULT_CONTENT_LAYERS,
        pooling: str = "max",
        weights: str = "keras",
        normalise: bool = True,
    ) -> None:
        super().__init__()
        self.arch = arch
        self.weights_source = weights
        features, names = build_vgg_features(arch, pooling)
        if weights == "keras":
            _load_keras_weights(features, names, arch)
        elif weights == "torchvision":
            _load_torchvision_weights(features, arch)
        else:
            raise ValueError("weights must be 'keras' or 'torchvision'")

        self.targets = set(layers)
        unknown = self.targets - set(names)
        if unknown:
            raise ValueError(f"Layers {sorted(unknown)} do not exist in {arch}")
        # Truncate the network after the deepest requested layer (saves compute).
        last = max(names.index(n) for n in self.targets)
        self.features = features[: last + 1].eval()
        self.names = names[: last + 1]
        for p in self.features.parameters():
            p.requires_grad_(False)

        if weights == "keras":
            # Caffe convention: BGR, 0-255, mean-subtracted.
            self.register_buffer("mean", torch.tensor([103.939, 116.779, 123.68]).view(1, 3, 1, 1))
            self.register_buffer("scale", torch.tensor(255.0))
        else:
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Gatys et al. (2016) rescale VGG so that every filter has unit mean
        # activation. We approximate this with fixed per-channel divisors
        # measured once on a calibration set of images (see calibrate()).
        self.normalise = normalise
        self.scales: Dict[str, torch.Tensor] = {}
        if normalise:
            self.scales = load_or_calibrate(self, arch, pooling, weights)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        if self.weights_source == "keras":
            return x.flip(1) * self.scale - self.mean
        return (x - self.mean) / self.std

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.preprocess(x)
        out: Dict[str, torch.Tensor] = {}
        for module, name in zip(self.features, self.names):
            x = module(x)
            if name in self.targets:
                out[name] = x / self.scales[name] if name in self.scales else x
        return out


CALIBRATION_DIRS = [Path(__file__).resolve().parent.parent / "data" / "content",
                    Path(__file__).resolve().parent.parent / "data" / "style"]


@torch.no_grad()
def load_or_calibrate(extractor: "VGGExtractor", arch: str, pooling: str, weights: str) -> Dict[str, torch.Tensor]:
    """Per-channel mean activation of every rectified conv layer on a fixed image set."""
    cache = WEIGHTS_DIR / f"norm_{arch}_{pooling}_{weights}.pt"
    if cache.exists():
        stored = torch.load(cache)
        return {k: v for k, v in stored.items() if k in extractor.targets}
    from .io import load_image

    files = sorted(p for d in CALIBRATION_DIRS for p in d.glob("*") if p.suffix.lower() in {".jpg", ".png"})
    full, names = build_vgg_features(arch, pooling)
    if weights == "keras":
        _load_keras_weights(full, names, arch)
    else:
        _load_torchvision_weights(full, arch)
    sums: Dict[str, torch.Tensor] = {}
    for f in files:
        x = extractor.preprocess(load_image(f, 256))
        for module, name in zip(full, names):
            x = module(x)
            if name.startswith("conv") and not name.endswith("_linear"):
                m = x.mean(dim=(0, 2, 3))
                sums[name] = sums.get(name, 0) + m / len(files)
    scales = {k: torch.where(v > 1e-6, v, torch.ones_like(v)).view(1, -1, 1, 1) for k, v in sums.items()}
    torch.save(scales, cache)
    return {k: v for k, v in scales.items() if k in extractor.targets}
