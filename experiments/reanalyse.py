"""Recompute metrics.json (and the per-run figures) of every finished run from its saved parameters.

Use after changing anything in src/actionable/analysis.py: no retraining is needed.
    python experiments/reanalyse.py [run_name ...]
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common                                                    # noqa: E402
from common import RUNS, load_cfg, load_params, read_json, write_json  # noqa: E402


def main(names=None):
    names = names or common.finished_runs()
    for n in names:
        d = RUNS / n
        if not (d / "params.npz").exists():
            print(f"[{n}] no params.npz -> skipped"); continue
        cfg = load_cfg(n)
        p = load_params(d / "params.npz")
        h = dict(np.load(d / "hist.npz")) if (d / "hist.npz").exists() else None
        old = read_json(d / "metrics.json") if (d / "metrics.json").exists() else {}
        m = common.analyse_params(p, cfg, h)
        for k in ("train_seconds_this_session",):
            if k in old:
                m[k] = old[k]
        write_json(d / "metrics.json", m)
        common.make_run_figures(n, p, cfg, m, h)
        print(f"[{n}] modules={m['n_modules']} hex={m.get('n_hex_modules', '-')}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
