"""Image loading, saving and figure helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from PIL import Image, ImageOps


def load_image(path: str | Path, long_side: int = 256, size: Optional[tuple] = None) -> torch.Tensor:
    """Load an RGB image as a (1, 3, H, W) tensor in [0, 1], keeping aspect ratio.

    ``size`` = (H, W) forces an exact size (used for style images and video frames).
    """
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if size is None:
        w, h = img.size
        scale = long_side / max(w, h)
        size = (max(1, round(h * scale)), max(1, round(w * scale)))
    img = img.resize((size[1], size[0]), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()


def to_pil(t: torch.Tensor) -> Image.Image:
    arr = (t.detach().cpu().squeeze(0).clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(arr)


def save_image(t: torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    to_pil(t).save(path)


def image_grid(rows: Sequence[Sequence[Image.Image]], row_labels=None, col_labels=None,
               cell: int = 220, path: Optional[str | Path] = None, title: Optional[str] = None,
               font_size: int = 11):
    """Render a labelled grid of PIL images with matplotlib and optionally save it."""
    import matplotlib.pyplot as plt

    n_r, n_c = len(rows), max(len(r) for r in rows)
    fig, axes = plt.subplots(n_r, n_c, figsize=(n_c * cell / 100, n_r * cell / 100 + (0.3 if title else 0)),
                             squeeze=False)
    for i, row in enumerate(rows):
        for j in range(n_c):
            ax = axes[i][j]
            ax.axis("off")
            if j < len(row) and row[j] is not None:
                ax.imshow(row[j])
            if i == 0 and col_labels:
                ax.set_title(col_labels[j], fontsize=font_size)
            if j == 0 and row_labels:
                ax.text(-0.04, 0.5, row_labels[i], transform=ax.transAxes, rotation=90,
                        va="center", ha="right", fontsize=font_size)
    if title:
        fig.suptitle(title, fontsize=font_size + 2)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
