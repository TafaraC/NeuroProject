"""Shared helpers for the experiment scripts (paths, run I/O, per-run analysis)."""
from __future__ import annotations
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from actionable import TrainConfig, analysis, plotting          # noqa: E402

# Results live in <repo>/results unless ACTIONABLE_RESULTS points elsewhere (run_all.py --quick uses results_quick).
RESULTS = Path(os.environ.get("ACTIONABLE_RESULTS", ROOT / "results"))
RUNS = RESULTS / "runs"
FIGS = RESULTS / "figures"
TABLES = RESULTS / "tables"


def ensure_dirs():
    for d in (RUNS, FIGS, TABLES):
        d.mkdir(parents=True, exist_ok=True)


class _Enc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return super().default(o)


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, cls=_Enc)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def run_dir(name: str) -> Path:
    d = RUNS / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_params(path):
    z = np.load(path)
    return {k: z[k] for k in ("a0", "A", "B", "K")}


def load_cfg(name: str) -> TrainConfig:
    c = read_json(RUNS / name / "config.json")
    known = set(asdict(TrainConfig()))
    return TrainConfig(**{k: v for k, v in c.items() if k in known})


def finished_runs():
    """Names of runs that have trained parameters and metrics on disk."""
    if not RUNS.exists():
        return []
    return sorted(d.name for d in RUNS.iterdir() if (d / "params.npz").exists() and (d / "metrics.json").exists())


# --------------------------------------------------------------------------- per-run analysis
def analyse_params(p, cfg: TrainConfig, hist=None, half_width: float = 2.5, res: int = 64, empirical: bool = True):
    """All the numbers we report for one trained code.  See README / report for definitions.

    * modules   : hierarchical clustering of the neurons' power spectra (analysis.cluster_modules)
    * lattice   : per-module peaks of the analytic autocorrelogram (analysis.lattice_summary)
    * gridness  : Sargolini score, per neuron, (i) analytic (infinite plane, own spectrum) and
                  (ii) empirical on the rate map of a 5 x 5 window (the usual experimentalist's way)
    * actionability : rank / error of the best linear action (analysis.actionability_report)
    """
    N, D, dim = p["a0"].shape[0], p["K"].shape[0], p["K"].shape[1]
    out = dict(N=int(N), D=int(D), dim=int(dim))
    norms = analysis.norms_from_samples(p, L=cfg.L)
    out["norm_rms_mean"] = float(norms.mean())
    out["norm_rms_min"] = float(norms.min())
    out["norm_rms_max"] = float(norms.max())
    labels = analysis.cluster_modules(p)
    out["labels"] = labels.tolist()
    ids = [m for m in sorted(set(labels.tolist())) if m >= 0]
    out["n_modules"] = len(ids)
    out["frac_in_modules"] = float(np.mean(labels >= 0))
    if hist is not None and len(hist.get("step", [])):
        for k in ("Lf", "Lp", "Lb", "lam_p", "lam_n"):
            out["final_" + k] = float(np.mean(np.asarray(hist[k])[-5:]))
    if dim != 2:
        out["modules"] = []
        return out

    mods = []
    for m in ids:
        idx = np.where(labels == m)[0]
        ls = analysis.lattice_summary(p, idx)
        ls["module"] = int(m)
        ls["size"] = int(len(idx))
        ls["hexagonal"] = bool(analysis.is_hexagonal(ls))
        mods.append(ls)
    out["modules"] = mods
    out["n_hex_modules"] = int(sum(m["hexagonal"] for m in mods))
    hexm = sorted([m for m in mods if m["hexagonal"]], key=lambda m: m["ac_spacing"])
    out["hex_spacings"] = [float(m["ac_spacing"]) for m in hexm]
    out["hex_sizes"] = [int(m["size"]) for m in hexm]
    out["hex_spacing_ratios"] = [float(b["ac_spacing"] / a["ac_spacing"]) for a, b in zip(hexm[:-1], hexm[1:])]
    out["hex_orientations_deg"] = [float(m["ac_orientation_deg"]) for m in hexm]

    g_an, neff = analysis.neuron_stats(p)
    out["gridness_analytic"] = g_an.tolist()
    out["neff"] = neff.tolist()
    out["frac_grid_analytic"] = float(np.mean(g_an > 0.3))
    out["frac_grid_analytic_in_hex"] = float(np.mean([m["hexagonal"] for m in mods for _ in range(m["size"])] or [0.0])
                                             * out["frac_in_modules"])
    maps = analysis.rate_maps(p, half_width=half_width, res=res)
    out["neg_mass"] = float(np.maximum(-maps, 0).sum() / np.abs(maps).sum())
    out["min_rate_mean"] = float(maps.min(axis=(1, 2)).mean())
    if empirical:
        g_emp = np.full(N, np.nan)
        for i in range(N):
            try:
                g_emp[i] = analysis.gridness(maps[i])
            except Exception:
                pass
        out["gridness_windowed"] = g_emp.tolist()
        out["frac_grid_windowed"] = float(np.nanmean(g_emp > 0.3))
    out["actionability"] = analysis.actionability_report(p, L=cfg.L)
    return out


def make_run_figures(name: str, p, cfg: TrainConfig, metrics: dict, hist=None):
    """rate maps coloured by module, frequency space and training curves for a run."""
    d = run_dir(name)
    if cfg.dim != 2:
        return
    labels = np.asarray(metrics["labels"])
    maps = analysis.rate_maps(p)
    g = metrics.get("gridness_windowed") or metrics["gridness_analytic"]
    plotting.plot_rate_maps(maps, labels, np.asarray(g), ncols=8, max_cells=cfg.N,
                            title=f"{name}: rate maps (border colour = module, number = gridness)",
                            fname=str(d / "rate_maps.png"))
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    kmax = float(np.abs(p["K"]).max()) * 1.05
    plotting.plot_freq_space(p, analysis.freq_power(p), labels, ax=ax, kmax=kmax, title=name)
    fig.tight_layout(); fig.savefig(d / "freq_space.png", dpi=150); plt.close(fig)
    if hist is not None and len(hist.get("step", [])):
        plotting.plot_training(hist, str(d / "training.png"))
