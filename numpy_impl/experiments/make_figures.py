"""Compose the figures of the report from the finished runs.

    results/figures/fig_main.png         rate maps coloured by module + frequency space + gridness histogram (main run)
    results/figures/fig_seeds.png        every main run: frequency space + example cells
    results/figures/fig_ablation.png     baseline and the three ablations
    results/figures/fig_sigma.png        effect of sigma on the module structure
    results/figures/fig_predictions.png  exp3 (sharpness) + exp4 (module angle) + exp5 (sigma sweep) summary
    results/figures/fig_oned.png         1-D codes (copy of exp2)
and copies the ones used in the report to review/figures/.
"""
from __future__ import annotations
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common                                                    # noqa: E402
from common import ROOT, RUNS, FIGS, load_cfg, load_params, read_json     # noqa: E402
from actionable import analysis, plotting                         # noqa: E402
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib import cm                                         # noqa: E402
from matplotlib.colors import to_rgb                              # noqa: E402

COL = plotting.MODULE_COLORS


def load_any(name, snap=None):
    d = RUNS / name
    f = d / ("params.npz" if snap is None else f"snap_{snap}.npz")
    if not f.exists():
        return None
    p = load_params(f)
    cfg = load_cfg(name)
    mf = d / ("metrics.json" if snap is None else f"metrics_snap{snap}.json")
    if mf.exists():
        m = read_json(mf)
    else:
        m = common.analyse_params(p, cfg, empirical=False)
        common.write_json(mf, m) if snap is not None else None
    return p, cfg, m


def mosaic(maps, labels, ncols=8, pad=3, cell=64):
    """One RGB image holding all rate maps (viridis, each scaled to its own range) with module-coloured frames."""
    N = len(maps)
    order = np.argsort(np.where(labels < 0, 999, labels), kind="stable")
    nrows = int(np.ceil(N / ncols))
    S = cell + 2 * pad
    img = np.ones((nrows * S, ncols * S, 3))
    for j, i in enumerate(order):
        r, c = divmod(j, ncols)
        m = maps[i]
        m = (m - m.min()) / (np.ptp(m) + 1e-12)
        tile = cm.viridis(m.T[::-1])[..., :3]
        frame = np.ones((S, S, 3)) * (np.array(to_rgb(COL[labels[i] % len(COL)])) if labels[i] >= 0 else 0.6)
        frame[pad:pad + cell, pad:pad + cell] = tile
        img[r * S:(r + 1) * S, c * S:(c + 1) * S] = frame
    return img


def example_cells(m, k=6):
    labels = np.asarray(m["labels"])
    ids = sorted({x["module"]: x["size"] for x in m["modules"]}.items(), key=lambda t: -t[1])
    picks = []
    rounds = 0
    while len(picks) < k and rounds < 6 and ids:
        for mod, _ in ids:
            idx = np.where(labels == mod)[0]
            if rounds < len(idx) and len(picks) < k:
                picks.append(int(idx[rounds]))
        rounds += 1
    strays = np.where(labels < 0)[0]
    for i in strays:
        if len(picks) < k:
            picks.append(int(i))
    return picks[:k]


def small_rows(ax_rows, entries, ncell=6):
    """entries: list of (title, p, cfg, m); axes: list of rows each [freq-space, cell...]"""
    for axs, (title, p, cfg, m) in zip(ax_rows, entries):
        labels = np.asarray(m["labels"])
        P = analysis.freq_power(p)
        plotting.plot_freq_space(p, P, labels, ax=axs[0], kmax=float(np.abs(p["K"]).max()) * 1.05)
        axs[0].set_title(title, fontsize=6.5, pad=2)
        axs[0].tick_params(labelsize=5); axs[0].set_xlabel(""); axs[0].set_ylabel("")
        maps = analysis.rate_maps(p, res=48)
        for a, i in zip(axs[1:1 + ncell], example_cells(m, ncell)):
            a.imshow(maps[i].T[::-1], cmap="viridis", interpolation="nearest")
            a.set_xticks([]); a.set_yticks([])
            if labels[i] >= 0:
                for s in a.spines.values():
                    s.set_color(COL[labels[i] % len(COL)]); s.set_linewidth(2)


def fig_main():
    r = load_any("main_sigma0.2_seed0")
    if r is None:
        return
    p, cfg, m = r
    labels = np.asarray(m["labels"])
    maps = analysis.rate_maps(p)
    fig = plt.figure(figsize=(7.2, 3.3))
    gs = fig.add_gridspec(2, 3, width_ratios=[2.1, 1.0, 1.0], height_ratios=[1, 1], wspace=0.35, hspace=0.45)
    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(mosaic(maps, labels)); ax.axis("off")
    ax.set_title("(a) rate maps, one frame colour per module", fontsize=7.5)
    ax = fig.add_subplot(gs[:, 1])
    plotting.plot_freq_space(p, analysis.freq_power(p), labels, ax=ax, kmax=float(np.abs(p["K"]).max()) * 1.05)
    ax.set_title("(b) learned frequencies", fontsize=7.5); ax.tick_params(labelsize=6)
    ax = fig.add_subplot(gs[0, 2])
    g = np.asarray(m["gridness_analytic"])
    ax.hist(g[labels >= 0], bins=np.linspace(-1, 1.6, 27), color="gray")
    ax.axvline(0.3, color="r", lw=0.8)
    ax.set_title("(c) gridness (analytic)", fontsize=7.5); ax.tick_params(labelsize=6)
    ax = fig.add_subplot(gs[1, 2])
    h = dict(np.load(RUNS / "main_sigma0.2_seed0" / "hist.npz"))
    ax.semilogy(h["step"], h["Lf"], lw=0.8, label="$L_f$")
    ax.semilogy(h["step"], np.maximum(h["Lp"], 1e-9), lw=0.8, label="$L_{pos}$")
    ax.legend(fontsize=5.5); ax.set_xlabel("step", fontsize=6); ax.tick_params(labelsize=6)
    ax.set_title("(d) training", fontsize=7.5)
    fig.savefig(FIGS / "fig_main.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_seeds():
    names = [n for n in ("main_sigma0.2_seed0", "main_sigma0.2_seed1", "main_sigma0.2_seed2",
                         "main_sigma1.0_seed0", "main_sigma1.0_seed1") if (RUNS / n / "params.npz").exists()]
    if not names:
        return
    ents = []
    for n in names:
        p, cfg, m = load_any(n)
        ents.append((f"{n.replace('main_', '')}: {m['n_modules']} mod., {m['n_hex_modules']} hex", p, cfg, m))
    fig, axes = plt.subplots(len(ents), 7, figsize=(7.2, 1.25 * len(ents)),
                             gridspec_kw=dict(width_ratios=[1.3] + [1] * 6, wspace=0.08, hspace=0.35), squeeze=False)
    small_rows(axes, ents)
    fig.savefig(FIGS / "fig_seeds.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_ablation():
    ents = []
    r = load_any("main_sigma0.2_seed0", 50000)
    if r:
        ents.append(("baseline (50k steps)", *r))
    for lab, n in (("no non-negativity", "abl_no_positivity"), ("no functional term", "abl_no_functional"),
                   ("no actionability (D=96)", "abl_no_actionability")):
        r = load_any(n)
        if r:
            ents.append((lab, *r))
    if len(ents) < 2:
        return
    fig, axes = plt.subplots(len(ents), 7, figsize=(7.2, 1.25 * len(ents)),
                             gridspec_kw=dict(width_ratios=[1.3] + [1] * 6, wspace=0.08, hspace=0.35), squeeze=False)
    small_rows(axes, ents)
    fig.savefig(FIGS / "fig_ablation.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_sigma():
    ents = []
    for lab, n, snap in (("sigma=0.1", "sweep_sigma0.1_seed0", None), ("sigma=0.2", "main_sigma0.2_seed0", 50000),
                         ("sigma=0.4", "sweep_sigma0.4_seed0", None), ("sigma=1.0", "main_sigma1.0_seed0", 50000)):
        r = load_any(n, snap)
        if r:
            p, cfg, m = r
            ents.append((f"{lab}: {m['n_modules']} mod.", p, cfg, m))
    if len(ents) < 2:
        return
    fig, axes = plt.subplots(len(ents), 7, figsize=(7.2, 1.25 * len(ents)),
                             gridspec_kw=dict(width_ratios=[1.3] + [1] * 6, wspace=0.08, hspace=0.35), squeeze=False)
    small_rows(axes, ents)
    fig.savefig(FIGS / "fig_sigma.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_predictions():
    parts = [("exp5_sigma.png", "(a)"), ("exp3_sharpness.png", "(b)"), ("exp4_module_angle.png", "(c)")]
    imgs = [(plt.imread(FIGS / f), t) for f, t in parts if (FIGS / f).exists()]
    if not imgs:
        return
    fig, axes = plt.subplots(len(imgs), 1, figsize=(7.2, 2.7 * len(imgs)))
    axes = np.atleast_1d(axes)
    for a, (im, t) in zip(axes, imgs):
        a.imshow(im); a.axis("off"); a.text(0.0, 1.0, t, transform=a.transAxes, fontsize=9, va="bottom")
    fig.tight_layout(pad=0.3)
    fig.savefig(FIGS / "fig_predictions.png", dpi=170, bbox_inches="tight"); plt.close(fig)


def main():
    common.ensure_dirs()
    for f in (fig_main, fig_seeds, fig_ablation, fig_sigma, fig_predictions):
        try:
            f()
            print(f"make_figures: {f.__name__} ok")
        except Exception as e:                                   # keep going: one missing run must not stop the rest
            print(f"make_figures: {f.__name__} FAILED: {e!r}")
    if (FIGS / "exp2_1d.png").exists():
        shutil.copy(FIGS / "exp2_1d.png", FIGS / "fig_oned.png")
    dst = ROOT / "review" / "figures"
    if dst.exists() and common.RESULTS == ROOT / "results":
        for f in ("fig_main.png", "fig_seeds.png", "fig_ablation.png", "fig_sigma.png", "fig_predictions.png", "fig_oned.png"):
            if (FIGS / f).exists():
                shutil.copy(FIGS / f, dst / f)


if __name__ == "__main__":
    main()
