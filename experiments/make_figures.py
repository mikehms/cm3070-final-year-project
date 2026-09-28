"""Generate every figure and summary table used in the report from results/."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RES = ROOT / "results"
IMG = RES / "images"
FIG = RES / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3})

C = ROOT / "data" / "content"
S = ROOT / "data" / "style"
CONTENT_FILES = {"tubingen": C / "tubingen.jpg", "portrait": C / "me.jpg", "cat": C / "cat.jpg",
                 "coffee": C / "coffee.png", "house": C / "house_sample.png"}
STYLE_FILES = {"starry_night": S / "starry_night.jpg", "scream": S / "the_scream.jpg", "wave": S / "wave.jpg",
               "shipwreck": S / "shipwreck.jpg", "abstract": S / "abstract_sample.png"}
BLUE, ORANGE, GREEN, RED, PURPLE, GREY = "#2f6db5", "#e08a1e", "#3a9a52", "#c8413b", "#7b52a8", "#777777"


def runs():
    with open(RES / "runs.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ["runtime_s", "evals", "final_total", "final_content", "final_style", "final_tv", "ssim", "edge_f1",
                  "colour_w1", "texture_chi2", "eval_style", "tv"]:
            r[k] = float(r[k])
        r["cfg"] = json.loads(r["config"])
    return {r["run_id"]: r for r in rows}


def im(name):
    return Image.open(IMG / f"{name}.png")


def thumb(path, h=256):
    img = Image.open(path).convert("RGB")
    w = round(img.width * h / img.height)
    return img.resize((w, h), Image.LANCZOS)


def grid(images, titles, path, ncols, row_labels=None, cell=2.1, fontsize=8.5):
    n = len(images)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * cell, nrows * cell * 0.9), squeeze=False)
    for k, ax in enumerate(axes.flat):
        ax.axis("off")
        if k < n and images[k] is not None:
            ax.imshow(images[k])
            ax.set_title(titles[k], fontsize=fontsize)
    if row_labels:
        for i, lab in enumerate(row_labels):
            axes[i][0].text(-0.06, 0.5, lab, transform=axes[i][0].transAxes, rotation=90, va="center",
                            ha="right", fontsize=fontsize + 0.5)
    fig.tight_layout(pad=0.4)
    fig.savefig(FIG / path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def load_log(run_id):
    with open(RES / "logs" / f"{run_id}.csv") as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def write_table(name, header, rows):
    with open(FIG / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ------------------------------------------------------------------ figures
def fig_e1(R):
    sws = ["1e+02", "1e+03", "1e+04", "1e+05", "1e+06", "1e+07"]
    imgs, titles = [], []
    for pair in ["tubingen_starry_night", "portrait_wave"]:
        for sw in sws:
            rid = f"E1_style_weight__{pair}_sw{sw}"
            imgs.append(im(rid) if rid in R else None)
            titles.append(f"β = 10{str(int(sw[-2:])).translate(str.maketrans('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹'))}" if rid in R else "")
    grid(imgs, titles, "e1_style_weight_grid.png", 6, row_labels=["Tübingen × Starry Night", "Portrait × Great Wave"])

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.9))
    for pair, col, lab in [("tubingen_starry_night", BLUE, "Tübingen × Starry Night"),
                           ("portrait_wave", ORANGE, "Portrait × Great Wave")]:
        xs, rows = [], []
        for sw in sws:
            rid = f"E1_style_weight__{pair}_sw{sw}"
            if rid in R:
                xs.append(float(sw)); rows.append(R[rid])
        if not rows:
            continue
        axes[0].plot(xs, [r["ssim"] for r in rows], "o-", color=col, label=lab)
        axes[0].plot(xs, [r["edge_f1"] for r in rows], "s--", color=col, alpha=0.6)
        axes[1].plot(xs, [r["eval_style"] for r in rows], "o-", color=col, label=lab)
        axes[2].plot(xs, [r["texture_chi2"] for r in rows], "o-", color=col, label=lab)
    axes[0].set_title("Content preservation (higher = more)\nSSIM solid, edge F1 dashed")
    axes[1].set_title("Deep style distance (evaluator)\nlog10, lower = closer to style")
    axes[2].set_title("LBP texture distance to style\nlower = closer texture")
    for ax in axes:
        ax.set_xscale("log"); ax.set_xlabel("style weight β (α = 1)")
    axes[0].legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "e1_tradeoff.png", dpi=170)
    plt.close(fig)


def fig_e2(R):
    rows = []
    base = R.get("E1_style_weight__tubingen_starry_night_sw1e+05")
    ref = [("α=1, β=10^5 (baseline)", base)]
    for rid, lab in [("E2_ratio__cw10_sw1e+06", "α=10, β=10^6"), ("E2_ratio__cw100_sw1e+07", "α=100, β=10^7"),
                     ("E2_ratio__cw10_sw1e+05", "α=10, β=10^5"), ("E2_ratio__cw100_sw1e+05", "α=100, β=10^5")]:
        ref.append((lab, R.get(rid)))
    for lab, r in ref:
        if r:
            rows.append([lab, f"{r['cfg']['style_weight'] / r['cfg']['content_weight']:.0e}", f"{r['ssim']:.3f}",
                         f"{r['edge_f1']:.3f}", f"{r['eval_style']:.2f}", f"{r['texture_chi2']:.4f}"])
    write_table("e2_ratio", ["setting", "beta/alpha", "SSIM", "edge F1", "eval_style", "texture chi2"], rows)
    imgs = [im(r["run_id"]) for _, r in ref if r]
    grid(imgs, [lab for lab, r in ref if r], "e2_ratio_grid.png", len(imgs))


def fig_e3(R):
    runs_ = [("E3_optimiser__lbfgs", "L-BFGS", BLUE), ("E3_optimiser__adam_lr0.005", "Adam lr=0.005", GREEN),
             ("E3_optimiser__adam_lr0.02", "Adam lr=0.02", ORANGE), ("E3_optimiser__adam_lr0.05", "Adam lr=0.05", RED),
             ("E3_optimiser__sgd_lr0.5", "SGD-momentum lr=0.5", PURPLE)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    rows = []
    for rid, lab, col in runs_:
        if rid not in R:
            continue
        log = load_log(rid)
        ev = [r["eval"] for r in log]
        axes[0].plot(ev, [r["total"] for r in log], color=col, label=lab, lw=1.3)
        axes[1].plot([r["time_s"] for r in log], [r["total"] for r in log], color=col, label=lab, lw=1.3)
        tot = np.array([r["total"] for r in log])
        final = tot[-1]
        # evaluations needed to get within 5% of L-BFGS's final loss
        target = None
        if "E3_optimiser__lbfgs" in R:
            lb = load_log("E3_optimiser__lbfgs")[-1]["total"]
            hit = np.where(tot <= lb * 1.05)[0]
            target = int(hit[0] + 1) if len(hit) else None
        r = R[rid]
        at = {n: tot[min(n, len(tot)) - 1] for n in (100, 200, 600)}
        rows.append([lab, f"{at[100]:.2f}", f"{at[200]:.2f}", f"{final:.2f}", target if target else ">600",
                     f"{r['runtime_s']:.0f}", f"{r['ssim']:.3f}", f"{r['eval_style']:.2f}"])
    for ax in axes:
        ax.set_yscale("log"); ax.set_ylabel("total loss"); ax.set_ylim(3, 30)
    axes[0].set_xlabel("loss evaluations"); axes[1].set_xlabel("wall-clock time (s, 2-core CPU)")
    axes[0].set_title("Convergence per evaluation"); axes[1].set_title("Convergence per second")
    axes[0].legend(fontsize=7.5, frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "e3_optimiser_curves.png", dpi=170); plt.close(fig)
    write_table("e3_optimiser", ["optimiser", "loss@100", "loss@200", "loss@600", "evals to within 5% of L-BFGS final",
                                 "runtime s", "SSIM", "eval_style"], rows)
    # snapshot grid
    imgs, titles = [], []
    snaps = [25, 100, 200, 600]
    labels = []
    for rid, lab, _ in [runs_[0], runs_[2], runs_[4]]:
        if rid not in R:
            continue
        labels.append(lab)
        for s in snaps:
            p = IMG / f"{rid}__eval{s}.png"
            if s == 600 and not p.exists():
                p = IMG / f"{rid}.png"
            imgs.append(Image.open(p) if p.exists() else None)
            titles.append(f"{s} evaluations" if len(labels) == 1 else "")
    grid(imgs, titles, "e3_snapshots.png", len(snaps), row_labels=labels)


def fig_e4(R):
    ids = [("E4_init__content", "content init"), ("E4_init__style", "style init"),
           ("E4_init__noise", "noise init (seed 0)"), ("E4_init__noise_seed1", "noise init (seed 1)")]
    imgs, titles, rows = [], [], []
    for rid, lab in ids:
        if rid in R:
            imgs.append(im(rid)); titles.append(lab)
            r = R[rid]
            rows.append([lab, f"{r['final_total']:.2f}", f"{r['ssim']:.3f}", f"{r['edge_f1']:.3f}",
                         f"{r['eval_style']:.2f}", f"{r['tv']:.4f}"])
    grid(imgs, titles, "e4_init_grid.png", len(imgs))
    write_table("e4_init", ["initialisation", "final loss", "SSIM", "edge F1", "eval_style", "TV"], rows)
    if "E4_init__noise_seed1" in R and "E4_init__noise" in R:
        a = np.asarray(im("E4_init__noise"), dtype=float) / 255
        b = np.asarray(im("E4_init__noise_seed1"), dtype=float) / 255
        with open(FIG / "e4_seed_diff.txt", "w") as f:
            f.write(f"mean abs pixel difference between noise seeds: {np.abs(a - b).mean():.4f}\n")


def fig_e5(R):
    sets = [("L1", "conv1_1"), ("L12", "conv1_1–2_1"), ("L123", "conv1_1–3_1"), ("L12345", "conv1_1–5_1"),
            ("L345", "conv3_1–5_1")]
    imgs, titles, rows = [], [], []
    for k, lab in sets:
        rid = f"E5_style_layers__{k}"
        if rid in R:
            imgs.append(im(rid)); titles.append(f"style: {lab}")
            r = R[rid]
            rows.append([f"style {lab}", f"{r['ssim']:.3f}", f"{r['edge_f1']:.3f}", f"{r['eval_style']:.2f}",
                         f"{r['texture_chi2']:.4f}", f"{r['runtime_s']:.0f}"])
    for cl in ["conv2_2", "conv3_2", "conv4_2", "conv5_2"]:
        rid = f"E5_content_layer__{cl}"
        if rid in R:
            imgs.append(im(rid)); titles.append(f"content: {cl}")
            r = R[rid]
            rows.append([f"content {cl}", f"{r['ssim']:.3f}", f"{r['edge_f1']:.3f}", f"{r['eval_style']:.2f}",
                         f"{r['texture_chi2']:.4f}", f"{r['runtime_s']:.0f}"])
    imgs += [None] * (10 - len(imgs)) if len(imgs) < 10 else []
    titles += [""] * (10 - len(titles)) if len(titles) < 10 else []
    # 5 style variants on row 1, 4 content variants on row 2
    grid(imgs[:5] + imgs[5:9] + [None], titles[:5] + titles[5:9] + [""], "e5_layers_grid.png", 5)
    write_table("e5_layers", ["setting", "SSIM", "edge F1", "eval_style", "texture chi2", "runtime s"], rows)


def fig_e6(R):
    rows, imgs, titles = [], [], []
    for pair, lab in [("tubingen_starry_night", "Tübingen × Starry Night"), ("cat_scream", "Cat × Scream")]:
        for arch in ["vgg16", "vgg19"]:
            for pool in ["max", "avg"]:
                rid = f"E6_architecture__{pair}_{arch}_{pool}"
                if rid not in R:
                    continue
                r = R[rid]
                imgs.append(im(rid)); titles.append(f"{arch.upper()} {pool}-pool")
                rows.append([lab, arch.upper(), pool, f"{r['runtime_s']:.0f}", f"{r['ssim']:.3f}",
                             f"{r['edge_f1']:.3f}", f"{r['eval_style']:.2f}", f"{r['texture_chi2']:.4f}",
                             f"{r['tv']:.4f}"])
    grid(imgs, titles, "e6_arch_grid.png", 4, row_labels=["Tübingen × Starry Night", "Cat × Scream"][:int(np.ceil(len(imgs) / 4))])
    write_table("e6_architecture", ["pair", "arch", "pooling", "runtime s", "SSIM", "edge F1", "eval_style",
                                    "texture chi2", "TV"], rows)


def fig_e7(R):
    contents = ["tubingen", "portrait", "cat", "coffee"]
    styles = ["starry_night", "scream", "wave", "shipwreck"]
    cell = 1.9
    fig, axes = plt.subplots(5, 5, figsize=(5 * cell * 1.15, 5 * cell * 0.95))
    for ax in axes.flat:
        ax.axis("off")
    names = {"starry_night": "The Starry Night", "scream": "The Scream", "wave": "The Great Wave",
             "shipwreck": "Shipwreck of the Minotaur"}
    for j, s in enumerate(styles):
        axes[0][j + 1].imshow(thumb(STYLE_FILES[s])); axes[0][j + 1].set_title(names[s], fontsize=8)
    for i, c in enumerate(contents):
        axes[i + 1][0].imshow(thumb(CONTENT_FILES[c]))
        for j, s in enumerate(styles):
            rid = f"E7_portfolio__{c}_{s}"
            if rid in R:
                axes[i + 1][j + 1].imshow(im(rid))
    axes[0][0].text(0.5, 0.5, "content ↓\nstyle →", ha="center", va="center", fontsize=9)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "e7_portfolio.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    rows = []
    for c in contents:
        for s in styles:
            rid = f"E7_portfolio__{c}_{s}"
            if rid in R:
                r = R[rid]
                rows.append([c, s, f"{r['ssim']:.3f}", f"{r['edge_f1']:.3f}", f"{r['colour_w1']:.2f}",
                             f"{r['texture_chi2']:.4f}", f"{r['eval_style']:.2f}", f"{r['runtime_s']:.0f}"])
    write_table("e7_portfolio", ["content", "style", "SSIM", "edge F1", "colour W1", "texture chi2", "eval_style",
                                 "runtime s"], rows)
    # style-wise / content-wise means
    if rows:
        arr = {k: [] for k in styles}
        arrc = {k: [] for k in contents}
        for c, s, ss, e, cw, t, es, rt in rows:
            arr[s].append((float(ss), float(es), float(cw)))
            arrc[c].append((float(ss), float(es), float(cw)))
        summ = [["style", s, *[f"{np.mean([v[k] for v in arr[s]]):.3f}" for k in range(3)]] for s in styles if arr[s]]
        summ += [["content", c, *[f"{np.mean([v[k] for v in arrc[c]]):.3f}" for k in range(3)]] for c in contents if arrc[c]]
        write_table("e7_means", ["group", "name", "mean SSIM", "mean eval_style", "mean colour W1"], summ)
    if "E7_portfolio__house_abstract" in R:
        grid([thumb(CONTENT_FILES["house"]), thumb(STYLE_FILES["abstract"]), im("E7_portfolio__house_abstract")],
             ["content (prototype input)", "style (prototype input)", "final system output"],
             "e7_house_final.png", 3, cell=2.6)


def fig_e8(R):
    imgs, titles = [], []
    for pair, cn, sn in [("tubingen_starry_night", "tubingen", "starry_night"), ("portrait_scream", "portrait", "scream")]:
        imgs += [thumb(CONTENT_FILES[cn]), im(f"E7_portfolio__{cn}_{sn}") if f"E7_portfolio__{cn}_{sn}" in R else None,
                 im(f"E8_colour__{pair}_preserve") if f"E8_colour__{pair}_preserve" in R else None]
        titles += ["content", "standard transfer", "colour preserved"]
    grid(imgs, titles, "e8_colour.png", 3, cell=2.4)
    imgs, titles = [], []
    order = [("E7_portfolio__tubingen_starry_night", "100% Starry Night"),
             ("E8_multistyle__starry0.75_scream0.25", "75 / 25"),
             ("E8_multistyle__starry0.5_scream0.5", "50 / 50"),
             ("E8_multistyle__starry0.25_scream0.75", "25 / 75"),
             ("E7_portfolio__tubingen_scream", "100% The Scream")]
    for rid, lab in order:
        if rid in R:
            imgs.append(im(rid)); titles.append(lab)
    grid(imgs, titles, "e8_blend.png", len(imgs), cell=2.2)
    rows = []
    for rid, lab in [("E7_portfolio__tubingen_starry_night", "Tübingen, standard"),
                     ("E8_colour__tubingen_starry_night_preserve", "Tübingen, colour preserved"),
                     ("E7_portfolio__portrait_scream", "Portrait, standard"),
                     ("E8_colour__portrait_scream_preserve", "Portrait, colour preserved")]:
        if rid in R:
            r = R[rid]
            rows.append([lab, f"{r['ssim']:.3f}", f"{r['colour_w1']:.2f}", f"{r['texture_chi2']:.4f}"])
    write_table("e8_colour", ["run", "SSIM", "colour W1 to style", "texture chi2"], rows)
    # blend: distance to each style via evaluator
    try:
        from nst import load_image
        from nst.metrics import eval_style
        import torch  # noqa: F401
        sn = load_image(STYLE_FILES["starry_night"], 256)
        sc = load_image(STYLE_FILES["scream"], 256)
        brows = []
        for rid, lab in order:
            if rid in R:
                o = load_image(IMG / f"{rid}.png", 256)
                brows.append([lab, f"{eval_style(sn, o):.2f}", f"{eval_style(sc, o):.2f}"])
        write_table("e8_blend", ["mix (Starry / Scream)", "eval_style to Starry Night", "eval_style to Scream"], brows)
    except Exception as exc:  # pragma: no cover
        print("blend metric failed", exc)


def fig_e9(R):
    for rid, cn, sn in [("E9_highres__tubingen_starry_512", "tubingen", "starry_night"),
                        ("E9_highres__portrait_wave_512", "portrait", "wave")]:
        if rid in R:
            img = im(rid)
            fig, ax = plt.subplots(figsize=(img.width / 100, img.height / 100))
            ax.imshow(img); ax.axis("off")
            fig.savefig(FIG / f"{rid}.png", dpi=150, bbox_inches="tight", pad_inches=0)
            plt.close(fig)


def fig_default_loss(R):
    rid = "E1_style_weight__tubingen_starry_night_sw1e+05"
    if rid not in R:
        return
    log = load_log(rid)
    fig, ax = plt.subplots(figsize=(6.2, 3))
    ev = [r["eval"] for r in log]
    ax.plot(ev, [r["total"] for r in log], color="k", label="total")
    ax.plot(ev, [r["w_content"] for r in log], color=BLUE, label="α·content")
    ax.plot(ev, [r["w_style"] for r in log], color=ORANGE, label="β·style")
    ax.plot(ev, [10 * r["tv"] for r in log], color=GREY, label="γ·TV")
    ax.set_yscale("log"); ax.set_ylim(0.05, 300); ax.set_xlabel("loss evaluations"); ax.set_ylabel("weighted loss")
    ax.legend(frameon=False, ncol=4, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "default_loss_curve.png", dpi=170); plt.close(fig)


def fig_metric_corr(R):
    keys = ["ssim", "edge_f1", "colour_w1", "texture_chi2", "eval_style", "tv", "final_content", "final_style"]
    labels = ["SSIM", "edge F1", "colour W1", "LBP χ²", "eval_style", "TV", "content loss", "style loss"]
    rs = [r for r in R.values() if r["experiment"] in ("E1_style_weight", "E2_ratio", "E5_style_layers",
                                                       "E5_content_layer", "E7_portfolio", "E4_init", "E6_architecture")]
    if len(rs) < 5:
        return
    from scipy.stats import spearmanr
    M = np.array([[r[k] for k in keys] for r in rs])
    rho = spearmanr(M).correlation
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.grid(False)
    h = ax.imshow(rho, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(keys))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(keys))); ax.set_yticklabels(labels)
    for i in range(len(keys)):
        for j in range(len(keys)):
            ax.text(j, i, f"{rho[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(rho[i, j]) > 0.6 else "black")
    fig.colorbar(h, fraction=0.046)
    ax.set_title(f"Spearman correlation between metrics (n = {len(rs)} runs)")
    fig.tight_layout(); fig.savefig(FIG / "metric_correlation.png", dpi=170); plt.close(fig)
    write_table("metric_corr", ["metric"] + labels, [[labels[i]] + [f"{v:.2f}" for v in rho[i]] for i in range(len(keys))])


def fig_video():
    V = RES / "video"
    if not (V / "video_summary.csv").exists():
        return
    methods = [("independent", "Independent per frame"), ("warm", "Warm start"), ("temporal", "Warm start + temporal loss")]
    frames = [0, 3, 6, 9, 11]
    fig, axes = plt.subplots(3, len(frames), figsize=(len(frames) * 1.9, 3 * 1.75))
    for i, (m, lab) in enumerate(methods):
        for j, t in enumerate(frames):
            ax = axes[i][j]; ax.axis("off")
            ax.imshow(Image.open(V / f"{m}_{t:02d}.png"))
            if i == 0:
                ax.set_title(f"frame {t + 1}", fontsize=8)
        axes[i][0].text(-0.06, 0.5, lab, transform=axes[i][0].transAxes, rotation=90, va="center", ha="right", fontsize=7.5)
    fig.tight_layout(pad=0.3); fig.savefig(FIG / "video_frames.png", dpi=170, bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 2.8))
    for (m, lab), col in zip(methods, [RED, ORANGE, BLUE]):
        with open(V / f"{m}_per_frame_flicker.csv") as f:
            rows = list(csv.DictReader(f))
        ax.plot([int(r["transition"]) for r in rows], [float(r["mae"]) for r in rows], "o-", color=col, label=lab)
    ax.set_xlabel("frame transition"); ax.set_ylabel("flicker (MAE on overlap)")
    ax.legend(frameon=False, fontsize=7.5); fig.tight_layout()
    fig.savefig(FIG / "video_flicker.png", dpi=170); plt.close(fig)


def fig_composites(R):
    proto = RES / "prototype" / "demo_output.png"
    if proto.exists() and "E7_portfolio__house_abstract" in R:
        grid([thumb(CONTENT_FILES["house"]), thumb(STYLE_FILES["abstract"]), thumb(proto),
              im("E7_portfolio__house_abstract")],
             ["content", "style", "preliminary prototype", "final system"], "prototype_vs_final.png", 4, cell=2.4)
    hr = [f"E9_highres__{k}" for k in ("tubingen_starry_512", "portrait_wave_512")]
    if all(k in R for k in hr):
        a, b = im(hr[0]), im(hr[1])
        h = 512
        a = a.resize((round(a.width * h / a.height), h)); b = b.resize((round(b.width * h / b.height), h))
        canvas = Image.new("RGB", (a.width + b.width + 12, h), "white")
        canvas.paste(a, (0, 0)); canvas.paste(b, (a.width + 12, 0))
        canvas.save(FIG / "highres_pair.png")
    parts = [FIG / "e8_colour.png", FIG / "e8_blend.png"]
    if all(p.exists() for p in parts):
        ims = [Image.open(p).convert("RGB") for p in parts]
        w = max(i.width for i in ims)
        ims = [i.resize((w, round(i.height * w / i.width))) for i in ims]
        canvas = Image.new("RGB", (w, sum(i.height for i in ims) + 20), "white")
        y = 0
        for i in ims:
            canvas.paste(i, (0, y)); y += i.height + 20
        canvas.save(FIG / "e8_extensions.png")


if __name__ == "__main__":
    R = runs()
    for fn in [fig_e1, fig_e2, fig_e3, fig_e4, fig_e5, fig_e6, fig_e7, fig_e8, fig_e9, fig_default_loss, fig_metric_corr]:
        try:
            fn(R)
        except Exception as exc:
            print(fn.__name__, "failed:", repr(exc))
    fig_video()
    fig_composites(R)
    print("figures written to", FIG)
