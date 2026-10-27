"""Experiment 5 - ablations (Appendix I, Fig. 15) and the effect of sigma and l (Appendix B.4, Figs. 8-9).

(A) Ablations at the 50k-step budget, sigma = 0.2: remove non-negativity / the functional term / actionability
    (D = 96 frequencies > N/2, so the code cannot be actionable) and compare with the nine baseline seeds
    (seed 0 = the 50k-step snapshot of main_sigma0.2_seed0; seeds 1-8 = base50_*).
(B) Number of modules vs sigma (Fig. 9 of the paper; N = 64) at l = 0.5 (our default) and l = 0.2 (the paper's Fig. 9 value).
(C) Number of modules / hexagonality vs l at sigma = 0.4 (Fig. 8 of the paper), and N = 100 at l = 0.1.

Output: results/tables/exp5_ablations_sigma.json, results/figures/exp5_*.png
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common                                                    # noqa: E402
from common import RUNS, FIGS, TABLES, load_cfg, load_params, read_json, write_json   # noqa: E402
from actionable import analysis                                  # noqa: E402

BASE_STEPS = 50_000


def metrics_for(name, snap=None):
    """metrics.json of a run, or (snap=step) the analysis of its saved snapshot at that step (cached)."""
    d = RUNS / name
    if snap is None:
        f = d / "metrics.json"
        return read_json(f) if f.exists() else None
    f = d / f"metrics_snap{snap}.json"
    if f.exists():
        return read_json(f)
    s = d / f"snap_{snap}.npz"
    if not s.exists():
        return None
    p = load_params(s)
    m = common.analyse_params(p, load_cfg(name))
    write_json(f, m)
    return m


def params_for(name, snap=None):
    f = RUNS / name / ("params.npz" if snap is None else f"snap_{snap}.npz")
    return load_params(f) if f.exists() else None


def const_share(p, cfg):
    """share of each neuron's mean-square rate carried by the constant (DC) coefficient a0 (eq. 8's ||a0||^2 / N)."""
    sd = analysis.norms_from_samples(p, L=cfg.L)
    return float(np.mean(p["a0"].astype(float) ** 2 / sd ** 2))


def summarise(ms, ps, cfg):
    keys = ["n_modules", "n_hex_modules", "frac_in_modules", "frac_grid_analytic", "frac_grid_windowed",
            "neg_mass", "min_rate_mean"]
    out = {}
    for k in keys:
        v = np.array([m[k] for m in ms if k in m], float)
        out[k] = dict(mean=float(np.nanmean(v)), sd=float(np.nanstd(v)), values=v.tolist()) if len(v) else None
    neff = [np.mean(m["neff"]) for m in ms if "neff" in m]
    out["neff"] = dict(mean=float(np.mean(neff)), sd=float(np.std(neff)), values=neff) if neff else None
    cs = [const_share(p, cfg) for p in ps]
    out["const_share"] = dict(mean=float(np.mean(cs)), sd=float(np.std(cs)), values=cs)
    out["actionability_rel_error"] = dict(mean=float(np.mean([m["actionability"]["actionability_rel_error"] for m in ms])),
                                          values=[m["actionability"]["actionability_rel_error"] for m in ms])
    out["n_runs"] = len(ms)
    return out


def ablations():
    cfg = None
    base_m, base_p = [], []
    m0 = metrics_for("main_sigma0.2_seed0", BASE_STEPS)
    if m0:
        base_m.append(m0); base_p.append(params_for("main_sigma0.2_seed0", BASE_STEPS)); cfg = load_cfg("main_sigma0.2_seed0")
    for s in range(1, 9):
        n = f"base50_sigma0.2_seed{s}"
        m = metrics_for(n)
        if m:
            base_m.append(m); base_p.append(params_for(n)); cfg = load_cfg(n)
    res = {}
    if base_m:
        res["baseline"] = summarise(base_m, base_p, cfg)
    for key, name in (("no_positivity", "abl_no_positivity"), ("no_functional", "abl_no_functional"),
                      ("no_actionability", "abl_no_actionability")):
        m = metrics_for(name)
        if m:
            res[key] = summarise([m], [params_for(name)], load_cfg(name))
    return res


def sweep(prefixes, key_fn):
    """group finished runs by key (e.g. sigma) -> list of metrics"""
    groups = {}
    for n in common.finished_runs():
        for pre in prefixes:
            if n.startswith(pre):
                k = key_fn(n)
                if k is not None:
                    groups.setdefault(k, []).append((n, metrics_for(n)))
    return groups


def sigma_sweep():
    out = {}
    # l = 0.5 (our default): sweep_sigma*, base50 (sigma 0.2, 9 seeds incl. the 50k snapshot of main seed 0) and the 50k snapshot of main sigma 1.0
    g = {}
    for n in common.finished_runs():
        if n.startswith("sweep_sigma"):
            g.setdefault(float(n.split("_sigma")[1].split("_")[0]), []).append(metrics_for(n))
    m0 = metrics_for("main_sigma0.2_seed0", BASE_STEPS)
    for n in ["base50_sigma0.2_seed%d" % s for s in range(1, 9)]:
        if (RUNS / n / "metrics.json").exists():
            g.setdefault(0.2, []).append(metrics_for(n))
    if m0:
        g.setdefault(0.2, []).append(m0)
    m1 = metrics_for("main_sigma1.0_seed0", BASE_STEPS)
    if m1:
        g.setdefault(1.0, []).append(m1)
    out["l=0.5"] = {str(k): v for k, v in sorted(g.items())}
    # l = 0.2 (the paper's Fig. 9 value)
    g = {}
    for n in common.finished_runs():
        if n.startswith("sweepl02_sigma"):
            g.setdefault(float(n.split("_sigma")[1].split("_")[0]), []).append(metrics_for(n))
        if n.startswith("sens_l0.2_sigma0.2"):
            g.setdefault(0.2, []).append(metrics_for(n))
    out["l=0.2"] = {str(k): v for k, v in sorted(g.items())}
    return out


def l_sweep():
    g = {}
    for n, l_ in (("sweepl_l1.0_sigma0.4", 1.0), ("sweep_sigma0.4_seed0", 0.5), ("sweep_sigma0.4_seed1", 0.5),
                  ("sweepl_l0.3_sigma0.4", 0.3), ("sweepl_l0.2_sigma0.4", 0.2), ("sweepl_l0.1_sigma0.4", 0.1)):
        m = metrics_for(n)
        if m:
            g.setdefault(l_, []).append(m)
    m = metrics_for("sweepl_l0.1_sigma0.4_N100")
    return dict(N64={str(k): v for k, v in sorted(g.items())}, N100_l0p1=m)


def stat(ms, k):
    v = np.array([m[k] for m in ms], float)
    return float(v.mean()), float(v.std()), v.tolist()


def main():
    common.ensure_dirs()
    res = dict(ablations=ablations(), sigma=sigma_sweep(), l=l_sweep())
    # compact tables for the report
    tab = {}
    for lk, d in res["sigma"].items():
        tab[lk] = {s: dict(n=len(ms), n_modules=stat(ms, "n_modules"), n_hex=stat(ms, "n_hex_modules"),
                           frac_in_modules=stat(ms, "frac_in_modules")[0])
                   for s, ms in d.items() if ms}
    res["sigma_table"] = tab
    write_json(TABLES / "exp5_ablations_sigma.json", res)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6))
    for lk, c in (("l=0.5", "C0"), ("l=0.2", "C3")):
        d = tab.get(lk, {})
        if not d:
            continue
        sg = np.array([float(s) for s in d]); mu = np.array([d[s]["n_modules"][0] for s in d]); sd = np.array([d[s]["n_modules"][1] for s in d])
        mh = np.array([d[s]["n_hex"][0] for s in d]); sh = np.array([d[s]["n_hex"][1] for s in d])
        ax[0].errorbar(sg, mu, sd, color=c, marker="o", ms=3, capsize=2, label=lk)
        ax[1].errorbar(sg, mh, sh, color=c, marker="o", ms=3, capsize=2, label=lk)
    for a, t in zip(ax, ("modules found", "hexagonal modules found")):
        a.set_xscale("log"); a.set_xlabel(r"neural length-scale $\sigma$"); a.set_ylabel(t); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIGS / "exp5_sigma.png", dpi=170); plt.close(fig)
    ab = res["ablations"]
    for k, v in ab.items():
        print(f"exp5 ablation {k:17s}: modules {v['n_modules']['mean']:.1f}, hexagonal {v['n_hex_modules']['mean']:.1f}, "
              f"gridness>0.3 {100 * v['frac_grid_analytic']['mean']:.0f}%, mean neff {v['neff']['mean']:.1f}, "
              f"a0 share {v['const_share']['mean']:.2f}, neg mass {v['neg_mass']['mean']:.3f}")
    for lk, d in tab.items():
        print(f"exp5 sigma sweep ({lk}): " + ", ".join(f"sigma={s}: {v['n_modules'][0]:.1f} modules/{v['n_hex'][0]:.1f} hex (n={v['n']})" for s, v in d.items()))


if __name__ == "__main__":
    main()
