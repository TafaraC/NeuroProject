"""Experiment 6 - sensitivity to the choices the paper leaves unspecified (our reproducibility audit).

Each row changes ONE thing relative to the baseline of the same sigma at the same 50k-step budget:
  * k_n = -4 instead of +4 (the paper's norm-constraint GECO target: with +4 the norm multiplier decays to its floor),
  * D = 20 frequencies instead of 31 (the paper does not give D),
  * initial frequency scale 2 or 8 instead of 4 (not given),
  * norm_mode 'mean' (mean-square normalisation over the M points) instead of 'sum' (eq. 22 read literally),
  * l = 0.2 instead of 0.5.
Baselines: sigma = 0.2 -> nine seeds (seed 0 = 50k snapshot of main_sigma0.2_seed0); sigma = 1.0 -> the 50k snapshot of main_sigma1.0_seed0.

Output: results/tables/exp6_sensitivity.json
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common                                                    # noqa: E402
from common import RUNS, TABLES, load_cfg, read_json, write_json   # noqa: E402
from exp5_ablations_sigma import metrics_for, BASE_STEPS          # noqa: E402

KEYS = ["n_modules", "n_hex_modules", "frac_in_modules", "frac_grid_analytic", "frac_grid_windowed"]

ROWS = [  # (label, run names, baseline key)
    ("k_n = -4", ["sens_kn-4_sigma0.2"], "sigma0.2"),
    ("k_n = -4", ["sens_kn-4_sigma1.0"], "sigma1.0"),
    ("D = 20", ["sens_D20"], "sigma0.2"),
    ("init freq sd 2", ["sens_freqstd2"], "sigma0.2"),
    ("init freq sd 8", ["sens_freqstd8"], "sigma0.2"),
    ("norm = mean", ["sens_normmean_sigma0.2"], "sigma0.2"),
    ("norm = mean", ["sens_normmean_sigma1.0"], "sigma1.0"),
    ("l = 0.2", ["sens_l0.2_sigma0.2_seed0", "sens_l0.2_sigma0.2_seed1"], "sigma0.2"),
]


def pack(ms):
    out = {}
    for k in KEYS:
        v = np.array([m[k] for m in ms if k in m], float)
        out[k] = dict(mean=float(np.nanmean(v)), sd=float(np.nanstd(v)), values=v.tolist()) if len(v) else None
    out["n_runs"] = len(ms)
    return out


def main():
    common.ensure_dirs()
    base = {"sigma0.2": [], "sigma1.0": []}
    m = metrics_for("main_sigma0.2_seed0", BASE_STEPS)
    if m:
        base["sigma0.2"].append(m)
    for s in range(1, 9):
        m = metrics_for(f"base50_sigma0.2_seed{s}")
        if m:
            base["sigma0.2"].append(m)
    m = metrics_for("main_sigma1.0_seed0", BASE_STEPS)
    if m:
        base["sigma1.0"].append(m)
    res = dict(baseline={k: pack(v) for k, v in base.items() if v}, rows=[])
    for label, names, bk in ROWS:
        ms = [metrics_for(n) for n in names if (RUNS / n / "metrics.json").exists()]
        if not ms:
            continue
        res["rows"].append(dict(label=label, runs=names, baseline=bk, **pack(ms)))
    write_json(TABLES / "exp6_sensitivity.json", res)
    for bk, v in res["baseline"].items():
        print(f"exp6 baseline {bk}: modules {v['n_modules']['mean']:.1f}+-{v['n_modules']['sd']:.1f}, "
              f"hex {v['n_hex_modules']['mean']:.1f}, gridness>0.3 {100 * v['frac_grid_analytic']['mean']:.0f}% (n={v['n_runs']})")
    for r in res["rows"]:
        print(f"exp6 {r['label']:16s} ({r['baseline']}): modules {r['n_modules']['mean']:.1f}, hex {r['n_hex_modules']['mean']:.1f}, "
              f"gridness>0.3 {100 * r['frac_grid_analytic']['mean']:.0f}%")


if __name__ == "__main__":
    main()
