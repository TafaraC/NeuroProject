"""Plot helpers (matplotlib only)."""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODULE_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2", "#17becf"]


def plot_rate_maps(maps, labels=None, scores=None, ncols=8, max_cells=64, title=None, fname=None, order_by_module=True):
    N = min(len(maps), max_cells)
    idx = np.arange(N)
    if labels is not None and order_by_module:
        key = np.where(labels[:N] < 0, 999, labels[:N])
        idx = np.argsort(key, kind="stable")
    nrows = int(np.ceil(N / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.25 * ncols, 1.25 * nrows))
    axes = np.atleast_2d(axes)
    for a in axes.ravel():
        a.axis("off")
    for j, i in enumerate(idx):
        a = axes.ravel()[j]
        a.imshow(maps[i].T, origin="lower", cmap="viridis", interpolation="nearest")
        if labels is not None and labels[i] >= 0:
            c = MODULE_COLORS[labels[i] % len(MODULE_COLORS)]
            for s in a.spines.values():
                s.set_visible(True); s.set_color(c); s.set_linewidth(2.5)
            a.axis("on"); a.set_xticks([]); a.set_yticks([])
        if scores is not None:
            a.set_title(f"{scores[i]:.2f}", fontsize=6, pad=1)
    if title:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    if fname:
        fig.savefig(fname, dpi=150)
        plt.close(fig)
    return fig


def plot_freq_space(p, power, labels=None, ax=None, kmax=None, title=None):
    """Dot plot of learned frequencies (and their mirror images); dot area ~ share of the module's spectral power."""
    K = p["K"]
    if ax is None:
        _, ax = plt.subplots(figsize=(3, 3))
    mods = [-1] if labels is None else sorted(set(labels.tolist()))
    for m in mods:
        idx = np.arange(power.shape[1]) if labels is None else np.where(labels == m)[0]
        if len(idx) == 0:
            continue
        pw = power[:, idx].mean(1)
        c = "gray" if m < 0 else MODULE_COLORS[m % len(MODULE_COLORS)]
        for sgn in (1, -1):
            ax.scatter(sgn * K[:, 0], sgn * K[:, 1], s=450 * pw / max(pw.sum(), 1e-9), color=c, alpha=0.6, linewidths=0)
    ax.axhline(0, color="k", lw=0.4); ax.axvline(0, color="k", lw=0.4)
    ax.set_aspect("equal")
    if kmax:
        ax.set_xlim(-kmax, kmax); ax.set_ylim(-kmax, kmax)
    ax.set_xlabel("$k_x$"); ax.set_ylabel("$k_y$")
    if title:
        ax.set_title(title, fontsize=8)
    return ax


def plot_training(hist, fname=None):
    fig, axs = plt.subplots(1, 3, figsize=(9, 2.5))
    axs[0].semilogy(hist["step"], np.maximum(hist["Lf"], 1e-12)); axs[0].set_title("functional loss $L_f$")
    axs[1].semilogy(hist["step"], np.maximum(hist["Lp"], 1e-12), label="$L_{pos}$")
    axs[1].semilogy(hist["step"], np.maximum(hist["Lb"], 1e-12), label="$L_{bound}$"); axs[1].legend(); axs[1].set_title("constraints")
    axs[2].semilogy(hist["step"], hist["lam_p"], label="$\\lambda_p$"); axs[2].semilogy(hist["step"], hist["lam_n"], label="$\\lambda_n$"); axs[2].legend(); axs[2].set_title("GECO weights")
    for a in axs:
        a.set_xlabel("step")
    fig.tight_layout()
    if fname:
        fig.savefig(fname, dpi=130); plt.close(fig)
    return fig
