#!/usr/bin/env python3
"""Draw the one figure of the review that needs both implementations: rate maps under a loose and a tight norm target.

    python make_figures.py        (needs numpy, matplotlib, jax; run from anywhere)

Rows (first 6 of 64 neurons each, same window [-1.5, 1.5]^2, per-panel colour scale):
  a  NumPy, sigma = 1.0, literal Table B.3 norm target (k_n = +4, inactive), final 150k-step run  (numpy_impl main_sigma1.0_seed0)
  b  NumPy, sigma = 1.0, tight norm target (k_n = -4),                       50k-step run         (numpy_impl sens_kn-4_sigma1.0)
  c  JAX,   sigma = 0.4, tight norm target 0.02,                             150k-step run, seed 0 (jax_impl diagnostic_study baseline_seed0)
  d  JAX,   sigma = 0.4, literal norm target exp(4),                         150k-step run, seed 0 (jax_impl diagnostic_study norm_table_seed0)
Output: figures/norm_target_montage.png
"""
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "numpy_impl" / "src"))
sys.path.insert(0, str(ROOT / "jax_impl"))
from actionable import analysis as np_analysis                         # noqa: E402
import jax.numpy as jnp                                                 # noqa: E402
from src.model import responses                                         # noqa: E402
from src.training import load_checkpoint                                # noqa: E402

EXT, RES, NCELL = 1.5, 340, 6        # RES >= 4*EXT*max|k|/pi+1 so that no learned frequency is aliased (JAX baseline: 261)


def numpy_maps(run):
    z = np.load(ROOT / "numpy_impl" / "results" / "runs" / run / "params.npz")
    p = {k: z[k] for k in ("a0", "A", "B", "K")}
    return np_analysis.rate_maps(p, half_width=EXT, res=RES)


def jax_maps(path):
    state, cfg = load_checkpoint(path)
    w, f = np.asarray(state.weights), np.asarray(state.frequencies)
    rng = np.random.default_rng(12345)                                  # same calibration as jax_impl/src/analysis.py
    calib = rng.normal(0, cfg.occupancy_scale, (cfg.samples, 2))
    norm = np.sqrt(np.maximum((np.asarray(responses(w, f, jnp.asarray(calib))) ** 2).sum(0), 1e-12))
    ax = np.linspace(-EXT, EXT, RES)
    x, y = np.meshgrid(ax, ax)
    pts = np.column_stack((x.ravel(), y.ravel()))
    return (np.asarray(responses(w, f, jnp.asarray(pts))) / norm).T.reshape(cfg.neurons, RES, RES)


jr = ROOT / "jax_impl" / "results" / "diagnostic_study" / "runs"
rows = [
    ("a  NumPy, $\\sigma$=1.0\nliteral target (inactive)\n150k steps", numpy_maps("main_sigma1.0_seed0")),
    ("b  NumPy, $\\sigma$=1.0\ntight target $k_n$=$-4$\n50k steps", numpy_maps("sens_kn-4_sigma1.0")),
    ("c  JAX, $\\sigma$=0.4\ntight target 0.02\n150k steps", jax_maps(jr / "baseline_seed0" / "checkpoint.npz")),
    ("d  JAX, $\\sigma$=0.4\nliteral target $e^{4}$\n150k steps", jax_maps(jr / "norm_table_seed0" / "checkpoint.npz")),
]
fig, axes = plt.subplots(len(rows), NCELL, figsize=(7.4, 5.0))
for r, (label, maps) in enumerate(rows):
    for c in range(NCELL):
        ax = axes[r, c]
        ax.imshow(maps[c], origin="lower", cmap="viridis", extent=[-EXT, EXT, -EXT, EXT], interpolation="bilinear")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.6)
    axes[r, 0].set_ylabel(label, rotation=0, ha="right", va="center", fontsize=7, labelpad=6)
fig.suptitle("Neurons 0-5 of the trained populations (64 neurons each); colour scale differs between panels", fontsize=8)
fig.tight_layout(rect=(0.0, 0.0, 1, 0.96))
out = Path(__file__).resolve().parent / "figures" / "norm_target_montage.png"
fig.savefig(out, dpi=170)
print("wrote", out, out.stat().st_size // 1024, "KB")
