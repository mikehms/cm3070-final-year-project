"""Video NST extension: frame-by-frame vs temporally-aware stylisation.

A synthetic camera pan across a still photograph gives a video whose optical
flow is known exactly (a horizontal shift of SHIFT pixels per frame). That
lets us warp the previous stylised frame without an optical-flow network and
measure flicker precisely. Three strategies are compared:

  independent : every frame stylised from its own content image (baseline)
  warm        : frame t is initialised with the warped stylised frame t-1
  temporal    : warm start + temporal consistency loss on the overlapping
                region (after Ruder, Dosovitskiy and Brox, 2016)
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from nst import NSTConfig, load_image, stylise, to_pil  # noqa: E402
from nst.metrics import evaluate  # noqa: E402

torch.set_num_threads(2)
OUT = ROOT / "results" / "video"
OUT.mkdir(parents=True, exist_ok=True)

N_FRAMES, SHIFT, H, W = 12, 8, 192, 224
STYLE = ROOT / "data" / "style" / "starry_night.jpg"
SOURCE = ROOT / "data" / "content" / "tubingen.jpg"


def make_frames():
    img = load_image(SOURCE, size=(252, 336))
    top = (252 - H) // 2
    return [img[:, :, top:top + H, t * SHIFT:t * SHIFT + W].contiguous() for t in range(N_FRAMES)]


def warp_prev(prev: torch.Tensor, cur_content: torch.Tensor):
    """Shift the previous output left by SHIFT px; the new strip comes from the content frame."""
    warped = cur_content.clone()
    warped[:, :, :, :W - SHIFT] = prev[:, :, :, SHIFT:]
    mask = torch.zeros_like(cur_content[:, :1])
    mask[:, :, :, :W - SHIFT] = 1.0
    return warped, mask


def flicker(outputs):
    """Mean absolute difference between consecutive outputs on the overlapping region."""
    d = [torch.mean(torch.abs(b[:, :, :, :W - SHIFT] - a[:, :, :, SHIFT:])).item()
         for a, b in zip(outputs[:-1], outputs[1:])]
    return float(np.mean(d)), d


def main():
    frames = make_frames()
    style = load_image(STYLE, 256)
    rows = []
    for method in ["independent", "warm", "temporal"]:
        outputs, t0 = [], time.perf_counter()
        for t, frame in enumerate(frames):
            if method == "independent" or t == 0:
                cfg = NSTConfig(steps=100, style_weight=1e5)
                res = stylise(frame, [style], cfg)
            else:
                init, mask = warp_prev(outputs[-1], frame)
                tw = 2e3 if method == "temporal" else 0.0
                cfg = NSTConfig(steps=40, style_weight=1e5, temporal_weight=tw)
                res = stylise(frame, [style], cfg, init_image=init, temporal_target=init, temporal_mask=mask)
            outputs.append(res.image)
            to_pil(res.image).save(OUT / f"{method}_{t:02d}.png")
        runtime = time.perf_counter() - t0
        fl, per = flicker(outputs)
        style_scores = [evaluate(f, style, o)["eval_style"] for f, o in zip(frames, outputs)]
        ssims = [evaluate(f, style, o)["ssim"] for f, o in zip(frames, outputs)]
        rows.append({"method": method, "runtime_s": round(runtime, 1), "flicker_mae": round(fl, 5),
                     "mean_eval_style": round(float(np.mean(style_scores)), 4),
                     "mean_ssim": round(float(np.mean(ssims)), 4)})
        with open(OUT / f"{method}_per_frame_flicker.csv", "w", newline="") as f:
            csv.writer(f).writerows([["transition", "mae"]] + [[i + 1, v] for i, v in enumerate(per)])
        pil = [to_pil(o).resize((W * 2, H * 2), Image.LANCZOS) for o in outputs]
        pil[0].save(OUT / f"{method}.gif", save_all=True, append_images=pil[1:], duration=120, loop=0)
        print(rows[-1], flush=True)
    raw = [to_pil(f).resize((W * 2, H * 2), Image.LANCZOS) for f in frames]
    raw[0].save(OUT / "content.gif", save_all=True, append_images=raw[1:], duration=120, loop=0)
    with open(OUT / "video_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
