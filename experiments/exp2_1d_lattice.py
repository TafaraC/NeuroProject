"""Experiment 2 - Section 3.1 of the paper: non-negativity turns a 1-D actionable code into a *lattice* code.

With the simplified loss L0 (eq. 7), bounded firing rates and non-negativity, the paper argues (Fig. 3B-C) that the
optimal neurons are periodic bump trains: sums of a base frequency and its HARMONICS, with the smallest possible
constant offset a0.  We test three predictions on the trained 1-D codes (runs oned_l0 and oned_l0_uniform):
  (a) the power sits on integer multiples of one fundamental frequency (harmonic lattice),
  (b) the neurons' tuning curves are non-negative bump trains,
  (c) harmonics reduce the mean rate relative to the RMS rate (a0 small) below the 0.816 of a single sinusoid + offset.

Output: results/tables/exp2_1d.json, results/figures/exp2_1d.png
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common                                                    # noqa: E402
from common import RUNS, FIGS, TABLES, load_cfg, load_params, write_json   # noqa: E402
from actionable import encode, analysis                           # noqa: E402

RUN_NAMES = ["oned_l0", "oned_l0_uniform"]


def analyse_1d(name):
    p = load_params(RUNS / name / "params.npz")
    cfg = load_cfg(name)
    k = np.abs(p["K"][:, 0]).astype(float)
    sd = analysis.norms_from_samples(p, L=cfg.L)
    P = ((p["A"] ** 2 + p["B"] ** 2) / sd ** 2).sum(1).astype(float)           # power per frequency (summed over cells)
    P_rel = P / P.sum()
    order = np.argsort(k)
    # fundamental = lowest frequency carrying >= 10 % of the strongest frequency's power
    strong = np.where(P >= 0.10 * P.max())[0]
    k1 = float(k[strong].min())
    ratio = k / k1
    dev = np.abs(ratio - np.round(ratio))
    # power-weighted deviation from an integer multiple (null: uniform ratios give 0.25)
    wdev = float((P_rel * dev).sum())
    # same statistic for all frequencies of the same set paired with a random fundamental (null distribution)
    rng = np.random.default_rng(0)
    null = []
    for _ in range(2000):
        kk = rng.choice(k, size=1)[0]
        null.append(float((P_rel * (np.abs(k / kk - np.round(k / kk)))).sum()))
    # tuning curves
    xs = np.linspace(-4.0, 4.0, 801)[:, None].astype(p["K"].dtype)
    G = encode(p, xs) / sd
    mean_over_rms = (G.mean(0) / np.sqrt((G ** 2).mean(0)))
    out = dict(run=name, N=int(cfg.N), D=int(cfg.D), occupancy=cfg.occupancy,
               fundamental=k1, freqs=k[order].tolist(), power_share=P_rel[order].tolist(), ratios=ratio[order].tolist(),
               power_weighted_deviation_from_integer=wdev,
               null_deviation_median=float(np.median(null)), null_deviation_p05=float(np.percentile(null, 5)),
               share_power_within_0p1_of_integer=float(P_rel[dev < 0.1].sum()),
               min_rate_over_rms=float((G.min(0)).mean()),
               frac_negative=float((G < -1e-3).mean()),
               mean_over_rms_mean=float(mean_over_rms.mean()), single_sinusoid_bound=float(np.sqrt(2.0 / 3.0)),
               n_effective_freqs=float(1.0 / np.sum(P_rel ** 2)))
    return out, xs[:, 0], G, k[order], P_rel[order], k1


def main():
    common.ensure_dirs()
    names = [n for n in RUN_NAMES if (RUNS / n / "params.npz").exists()]
    if not names:
        print("exp2: no 1-D runs found -> nothing to do"); return
    import matplotlib.pyplot as plt
    res = {}
    fig, axes = plt.subplots(len(names), 3, figsize=(9.2, 2.5 * len(names)), squeeze=False)
    for r, n in enumerate(names):
        out, xs, G, k, P, k1 = analyse_1d(n)
        res[n] = out
        ax = axes[r, 0]
        for i in range(0, G.shape[1], max(1, G.shape[1] // 6)):
            ax.plot(xs, G[:, i], lw=1)
        ax.set_title(f"{n}: tuning curves", fontsize=8); ax.set_xlabel("x"); ax.axhline(0, color="k", lw=0.4)
        ax = axes[r, 1]
        ax.stem(k, P, basefmt=" ")
        for m in range(1, int(k.max() / k1) + 2):
            ax.axvline(m * k1, color="r", lw=0.5, alpha=0.5)
        ax.set_title(f"power per frequency (red: multiples of k1={k1:.2f})", fontsize=8); ax.set_xlabel("|k|")
        ax = axes[r, 2]
        ax.hist(G.mean(0) / np.sqrt((G ** 2).mean(0)), bins=10, color="gray")
        ax.axvline(np.sqrt(2 / 3), color="r", lw=1)
        ax.set_title("mean/RMS rate per cell (red: sinusoid + offset)", fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS / "exp2_1d.png", dpi=170); plt.close(fig)
    write_json(TABLES / "exp2_1d.json", res)
    for n, o in res.items():
        print(f"{n}: fundamental {o['fundamental']:.2f}; power-weighted deviation from integer ratios "
              f"{o['power_weighted_deviation_from_integer']:.3f} (null median {o['null_deviation_median']:.3f}); "
              f"mean/RMS {o['mean_over_rms_mean']:.2f}; fraction negative {o['frac_negative']:.3f}")


if __name__ == "__main__":
    main()
