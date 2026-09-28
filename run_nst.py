"""Command-line interface for the NST system.

Example:
    python run_nst.py --content data/content/tubingen.jpg --style data/style/starry_night.jpg \
        --out outputs/tubingen_starry.png --style-weight 1e5 --optimizer lbfgs --steps 300

Multi-style blend:
    python run_nst.py --content data/content/tubingen.jpg \
        --style data/style/starry_night.jpg data/style/the_scream.jpg --style-mix 0.5 0.5 ...
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from nst import NSTConfig, evaluate, load_image, save_image, stylise, to_pil
from nst.io import image_grid


def main() -> None:
    p = argparse.ArgumentParser(description="Optimisation-based Neural Style Transfer")
    p.add_argument("--content", required=True)
    p.add_argument("--style", required=True, nargs="+")
    p.add_argument("--style-mix", type=float, nargs="+")
    p.add_argument("--out", default="outputs/stylised.png")
    p.add_argument("--image-size", type=int, default=256, help="long side in pixels")
    p.add_argument("--arch", choices=["vgg16", "vgg19"], default="vgg19")
    p.add_argument("--pooling", choices=["max", "avg"], default="max")
    p.add_argument("--weights", choices=["keras", "torchvision"], default="keras")
    p.add_argument("--content-layers", nargs="+", default=["conv4_2"])
    p.add_argument("--style-layers", nargs="+", default=["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"])
    p.add_argument("--content-weight", type=float, default=1.0)
    p.add_argument("--style-weight", type=float, default=1e5)
    p.add_argument("--tv-weight", type=float, default=10.0)
    p.add_argument("--optimizer", choices=["lbfgs", "adam", "sgd"], default="lbfgs")
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--init", choices=["content", "noise", "style"], default="content")
    p.add_argument("--preserve-colour", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    cfg = NSTConfig(arch=a.arch, pooling=a.pooling, weights=a.weights, content_layers=a.content_layers,
                    style_layers=a.style_layers, content_weight=a.content_weight, style_weight=a.style_weight,
                    tv_weight=a.tv_weight, optimizer=a.optimizer, lr=a.lr, steps=a.steps, init=a.init,
                    preserve_colour=a.preserve_colour, style_mix=a.style_mix, seed=a.seed)
    content = load_image(a.content, a.image_size)
    styles = [load_image(s, a.image_size) for s in a.style]

    def progress(i, row):
        if i == 1 or i % 25 == 0:
            print(f"eval {i:4d}/{cfg.steps}  total={row['total']:.3f}  content={row['w_content']:.3f}  "
                  f"style={row['w_style']:.3f}  tv={row['tv']:.4f}")

    res = stylise(content, styles, cfg, on_step=progress)
    out = Path(a.out)
    save_image(res.image, out)
    stem = out.with_suffix("")

    with open(f"{stem}_loss_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res.log[0].keys()))
        w.writeheader()
        w.writerows(res.log)

    plt.figure(figsize=(7, 4))
    for key, label in [("total", "total"), ("w_content", "alpha x content"), ("w_style", "beta x style")]:
        plt.plot([r["eval"] for r in res.log], [r[key] for r in res.log], label=label)
    plt.yscale("log"); plt.xlabel("loss evaluations"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
    plt.savefig(f"{stem}_loss_curve.png", dpi=150); plt.close()

    image_grid([[to_pil(content), to_pil(styles[0]), to_pil(res.image)]],
               col_labels=["Content", "Style", "Output"], path=f"{stem}_comparison_grid.png")

    metrics = evaluate(content, styles[0], res.image)
    meta = {"content": a.content, "style": a.style, "runtime_s": res.runtime_s, "evals": len(res.log),
            "metrics": metrics, "config": cfg.to_dict()}
    Path(f"{stem}_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved {out} in {res.runtime_s:.1f}s | metrics: " +
          ", ".join(f"{k}={v:.3f}" for k, v in metrics.items()))


if __name__ == "__main__":
    main()
