"""Colour-preservation helpers (luminance-only transfer, Gatys et al., 2017)."""
from __future__ import annotations

import torch

# RGB <-> YIQ
_RGB2YIQ = torch.tensor([[0.299, 0.587, 0.114],
                         [0.596, -0.274, -0.322],
                         [0.211, -0.523, 0.312]])
_YIQ2RGB = torch.linalg.inv(_RGB2YIQ)


def rgb_to_yiq(x: torch.Tensor) -> torch.Tensor:
    return torch.einsum("ij,bjhw->bihw", _RGB2YIQ.to(x), x)


def yiq_to_rgb(x: torch.Tensor) -> torch.Tensor:
    return torch.einsum("ij,bjhw->bihw", _YIQ2RGB.to(x), x)


def luminance_only(x: torch.Tensor) -> torch.Tensor:
    """Return the Y channel replicated to 3 channels so VGG sees a grey image."""
    y = rgb_to_yiq(x)[:, :1]
    return y.repeat(1, 3, 1, 1)


def match_luminance(style_grey: torch.Tensor, content_grey: torch.Tensor) -> torch.Tensor:
    """Match the mean/std of the style luminance to the content luminance."""
    s_mu, s_sd = style_grey.mean(), style_grey.std()
    c_mu, c_sd = content_grey.mean(), content_grey.std()
    return ((style_grey - s_mu) / (s_sd + 1e-8) * c_sd + c_mu).clamp(0, 1)


def recombine_luminance(stylised_grey: torch.Tensor, content_rgb: torch.Tensor) -> torch.Tensor:
    """Combine the stylised luminance with the content image's chroma (I, Q)."""
    y = stylised_grey.mean(1, keepdim=True)
    iq = rgb_to_yiq(content_rgb)[:, 1:]
    return yiq_to_rgb(torch.cat([y, iq], 1)).clamp(0, 1)
