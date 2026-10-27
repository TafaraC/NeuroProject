"""Experiment 3 - Section 4.1 / Appendix J: the number of neurons in a module scales as N ~ (nu / mu)^2.

nu = lattice length-scale (grid spacing, mean distance of the six peaks round the origin of the module's
     autocorrelogram), mu = field width (std of a Gaussian field; the autocorrelogram's central peak has FWHM
     sqrt(2) * 2.355 * mu).  Both are measured on every hexagonal module of every finished run; module size is the
     number of neurons in the cluster.  We regress log(size) on log(nu/mu) (prediction: slope 2) and report the
     rank correlation.  Modules from different runs are not independent (same run, same seed), so the slope's 95 %
     interval is a bootstrap over RUNS.

Output: results/tables/exp3_sharpness.json, results/figures/exp3_sharpness.png
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common                                                    # noqa: E402
from common import RUNS, FIGS, TABLES, load_cfg, load_params, read_json, write_json   # noqa: E402
from actionable import analysis                                  # noqa: E402

MIN_SIZE = 3


def module_rows(names):
    rows = []
    for n in names:
        m = read_json(RUNS / n / "metrics.json")
        if m.get("dim", 2) != 2 or not m.get("modules"):
            continue
        p = load_params(RUNS / n / "params.npz")
        labels = np.asarray(m["labels"])
        P_all = analysis.freq_power(p).astype(float)
        K = p["K"].astype(float)
        for mod in m["modules"]:
            if not mod["hexagonal"] or mod["size"] < MIN_SIZE:
                continue
            idx = np.where(labels == mod["module"])[0]
            P = P_all[:, idx].mean(1)
            nu = mod["ac_spacing"]
            fwhm_ac = analysis.central_peak_fwhm(K, P, nu)
            mu = fwhm_ac / (np.sqrt(2.0) * 2.3548)
            # number of frequencies carrying > 1 % of the module's power (the quantity the paper's argument counts)
            n_freq = int((P / P.sum() > 0.01).sum())
            rows.append(dict(run=n, N=int(m["N"]), size=int(mod["size"]), nu=float(nu), mu=float(mu),
                             ratio=float(nu / mu), n_freq=n_freq, gridness=float(mod["gridness_analytic"])))
    return rows


def fit(rows, n_boot=2000, seed=0):
    x = np.log([r["ratio"] for r in rows]); y = np.log([r["size"] for r in rows])
    slope, icpt, r, pval, se = stats.linregress(x, y)
    rho, prho = stats.spearmanr(x, y)
    runs = sorted(set(r_["run"] for r_ in rows))
    by = {u: [i for i, r_ in enumerate(rows) if r_["run"] == u] for u in runs}
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        pick = rng.choice(len(runs), size=len(runs), replace=True)
        idx = [i for j in pick for i in by[runs[j]]]
        if len(set(x[idx])) < 3:
            continue
        bs.append(stats.linregress(x[idx], y[idx]).slope)
    ci = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))) if bs else (float("nan"), float("nan"))
    # reverse regression (log ratio on log size): slope should be 1/2 if N ~ ratio^2
    rev = stats.linregress(y, x)
    return dict(n_modules=len(rows), n_runs=len(runs), slope=float(slope), slope_ci95=ci, slope_se_naive=float(se),
                intercept=float(icpt), r=float(r), spearman=float(rho), spearman_p=float(prho),
                reverse_slope=float(rev.slope), paper_slope=2.0)


def main():
    common.ensure_dirs()
    names = common.finished_runs()
    rows = module_rows(names)
    if len(rows) < 4:
        print(f"exp3: only {len(rows)} hexagonal modules found -> skipping the regression");
        write_json(TABLES / "exp3_sharpness.json", dict(rows=rows, fit=None)); return
    res = fit(rows)
    write_json(TABLES / "exp3_sharpness.json", dict(rows=rows, fit=res))
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.7))
    r = np.array([x["ratio"] for x in rows]); s = np.array([x["size"] for x in rows])
    Ns = np.array([x["N"] for x in rows])
    sc = ax[0].scatter(r, s, c=np.log2(Ns), cmap="viridis", s=22)
    xx = np.linspace(r.min(), r.max(), 50)
    ax[0].plot(xx, np.exp(res["intercept"]) * xx ** res["slope"], "k-", lw=1, label=f"fit: slope {res['slope']:.2f}")
    c0 = np.exp(np.mean(np.log(s) - 2 * np.log(r)))
    ax[0].plot(xx, c0 * xx ** 2, "r--", lw=1, label="paper: slope 2")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel(r"$\nu/\mu$ (spacing / field width)"); ax[0].set_ylabel("neurons in module")
    ax[0].legend(fontsize=7); plt.colorbar(sc, ax=ax[0]).set_label("log2 N of run", fontsize=7)
    nf = np.array([x["n_freq"] for x in rows])
    ax[1].scatter(r, nf, c=np.log2(Ns), cmap="viridis", s=22)
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel(r"$\nu/\mu$"); ax[1].set_ylabel("frequencies with >1% of module power")
    fig.tight_layout(); fig.savefig(FIGS / "exp3_sharpness.png", dpi=170); plt.close(fig)
    print(f"exp3: {res['n_modules']} hexagonal modules from {res['n_runs']} runs; slope of log(size) on log(nu/mu) = "
          f"{res['slope']:.2f} (95% bootstrap CI {res['slope_ci95'][0]:.2f}..{res['slope_ci95'][1]:.2f}; paper: 2); "
          f"Spearman {res['spearman']:.2f}")


if __name__ == "__main__":
    main()
