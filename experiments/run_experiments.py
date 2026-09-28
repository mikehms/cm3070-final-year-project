"""Run the full experimental study used in the final report.

Every run is logged to results/runs.csv (config + runtime + metrics) and each
run's per-evaluation loss log goes to results/logs/<run_id>.csv. Finished runs
are skipped, so the script can be interrupted and resumed.

Usage:  python experiments/run_experiments.py [experiment ...]
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nst import NSTConfig, evaluate, load_image, save_image, stylise  # noqa: E402

torch.set_num_threads(2)
RES = ROOT / "results"
(RES / "images").mkdir(parents=True, exist_ok=True)
(RES / "logs").mkdir(parents=True, exist_ok=True)
RUNS_CSV = RES / "runs.csv"
C = ROOT / "data" / "content"
S = ROOT / "data" / "style"

CONTENTS = {"tubingen": C / "tubingen.jpg", "portrait": C / "me.jpg",
            "cat": C / "cat.jpg", "coffee": C / "coffee.png", "house": C / "house_sample.png"}
STYLES = {"starry_night": S / "starry_night.jpg", "scream": S / "the_scream.jpg",
          "wave": S / "wave.jpg", "shipwreck": S / "shipwreck.jpg", "abstract": S / "abstract_sample.png"}

BASE = dict(optimizer="lbfgs", steps=200, style_weight=1e5, content_weight=1.0, tv_weight=10.0)
SIZE = 256


def done_ids() -> set:
    if not RUNS_CSV.exists():
        return set()
    with open(RUNS_CSV) as f:
        return {r["run_id"] for r in csv.DictReader(f)}


def record(row: dict) -> None:
    new = not RUNS_CSV.exists()
    with open(RUNS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def run(exp: str, run_id: str, content: str, style, size: int = SIZE, snapshot_at=(), **overrides):
    run_id = f"{exp}__{run_id}"
    if run_id in done_ids():
        return
    cfg = NSTConfig(**{**BASE, **overrides})
    c = load_image(CONTENTS[content], size)
    style_names = [style] if isinstance(style, str) else list(style)
    ss = [load_image(STYLES[n], size) for n in style_names]
    res = stylise(c, ss, cfg, snapshot_at=snapshot_at)
    out = RES / "images" / f"{run_id}.png"
    save_image(res.image, out)
    for k, snap in getattr(res, "snapshots", {}).items():
        save_image(snap, RES / "images" / f"{run_id}__eval{k}.png")
    with open(RES / "logs" / f"{run_id}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res.log[0].keys()))
        w.writeheader()
        w.writerows(res.log)
    metrics = evaluate(c, ss[0], res.image)
    last = res.log[-1]
    row = {"run_id": run_id, "experiment": exp, "content": content, "style": "+".join(style_names),
           "size": f"{c.shape[2]}x{c.shape[3]}", "runtime_s": round(res.runtime_s, 2), "evals": len(res.log),
           "final_total": last["total"], "final_content": last["content"], "final_style": last["style"],
           "final_tv": last["tv"], **{k: round(v, 5) for k, v in metrics.items()},
           "config": json.dumps(cfg.to_dict())}
    record(row)
    print(f"{run_id}: {res.runtime_s:.0f}s ssim={metrics['ssim']:.3f} eval_style={metrics['eval_style']:.2f}",
          flush=True)


# ---------------------------------------------------------------- experiments
def e1_style_weight():
    for pair in [("tubingen", "starry_night"), ("portrait", "wave")]:
        for sw in [1e2, 1e3, 1e4, 1e5, 1e6, 1e7]:
            run("E1_style_weight", f"{pair[0]}_{pair[1]}_sw{sw:.0e}", *pair, style_weight=sw)


def e2_ratio():
    # Same alpha/beta ratio at different absolute scales -> is only the ratio important?
    for cw, sw in [(10, 1e6), (100, 1e7), (10, 1e5), (100, 1e5)]:
        run("E2_ratio", f"cw{cw}_sw{sw:.0e}", "tubingen", "starry_night", content_weight=cw, style_weight=sw)


def e3_optimiser():
    snaps = (25, 50, 100, 200, 400, 600)
    run("E3_optimiser", "lbfgs", "tubingen", "starry_night", optimizer="lbfgs", steps=600, snapshot_at=snaps)
    for lr in [0.005, 0.02, 0.05]:
        run("E3_optimiser", f"adam_lr{lr}", "tubingen", "starry_night", optimizer="adam", lr=lr, steps=600,
            snapshot_at=snaps)
    run("E3_optimiser", "sgd_lr0.5", "tubingen", "starry_night", optimizer="sgd", lr=0.5, steps=600,
        snapshot_at=snaps)


def e4_init():
    for init in ["content", "style", "noise"]:
        run("E4_init", init, "tubingen", "starry_night", init=init, steps=400, snapshot_at=(50, 100, 200))
    run("E4_init", "noise_seed1", "tubingen", "starry_night", init="noise", steps=400, seed=1)


def e5_layers():
    sets = {"L1": ["conv1_1"], "L12": ["conv1_1", "conv2_1"], "L123": ["conv1_1", "conv2_1", "conv3_1"],
            "L12345": ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"],
            "L345": ["conv3_1", "conv4_1", "conv5_1"]}
    for k, layers in sets.items():
        run("E5_style_layers", k, "tubingen", "starry_night", style_layers=layers)
    for cl in ["conv2_2", "conv3_2", "conv4_2", "conv5_2"]:
        run("E5_content_layer", cl, "tubingen", "starry_night", content_layers=[cl])


def e6_architecture():
    for pair in [("tubingen", "starry_night"), ("cat", "scream")]:
        for arch in ["vgg16", "vgg19"]:
            for pool in ["max", "avg"]:
                run("E6_architecture", f"{pair[0]}_{pair[1]}_{arch}_{pool}", *pair, arch=arch, pooling=pool)


def e7_portfolio():
    for c in ["tubingen", "portrait", "cat", "coffee"]:
        for s in ["starry_night", "scream", "wave", "shipwreck"]:
            run("E7_portfolio", f"{c}_{s}", c, s)
    run("E7_portfolio", "house_abstract", "house", "abstract")


def e8_extensions():
    for pair in [("tubingen", "starry_night"), ("portrait", "scream")]:
        run("E8_colour", f"{pair[0]}_{pair[1]}_preserve", *pair, preserve_colour=True)
    for w in [0.25, 0.5, 0.75]:
        run("E8_multistyle", f"starry{w}_scream{1 - w}", "tubingen", ["starry_night", "scream"],
            style_mix=[w, 1 - w])


def e9_highres():
    run("E9_highres", "tubingen_starry_512", "tubingen", "starry_night", size=512, steps=300)
    run("E9_highres", "portrait_wave_512", "portrait", "wave", size=512, steps=300)


EXPERIMENTS = {"e1": e1_style_weight, "e2": e2_ratio, "e3": e3_optimiser, "e4": e4_init, "e5": e5_layers,
               "e6": e6_architecture, "e7": e7_portfolio, "e8": e8_extensions, "e9": e9_highres}

if __name__ == "__main__":
    for name in (sys.argv[1:] or list(EXPERIMENTS)):
        EXPERIMENTS[name]()
