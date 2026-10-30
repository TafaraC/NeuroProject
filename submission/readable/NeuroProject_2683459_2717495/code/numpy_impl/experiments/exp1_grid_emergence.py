"""Train ONE actionable code and analyse it.  This is the workhorse every other experiment builds on.

    python experiments/exp1_grid_emergence.py --job main_sigma0.2_seed0        # a job from jobs.py (resumes if interrupted)
    python experiments/exp1_grid_emergence.py --name my_run --sigma 0.3 --seed 5 --steps 20000   # ad-hoc run

Outputs (results/runs/<name>/): config.json, params.npz, hist.npz, metrics.json, rate_maps.png, freq_space.png,
training.png, train.log (when started by run_all.py) and, for jobs with snap_every>0, snap_<step>.npz.
"""
from __future__ import annotations
import argparse
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common                                                   # noqa: E402
from common import TrainConfig, run_dir, write_json              # noqa: E402
from actionable import train                                     # noqa: E402
from jobs import JOBS, quick_cfg                                  # noqa: E402


def run(name: str, cfg: TrainConfig, snap_every: int = 0, force: bool = False, verbose: bool = True):
    common.ensure_dirs()
    d = run_dir(name)
    if (d / "metrics.json").exists() and not force:
        print(f"[{name}] already finished -> skipping (use --force to redo)", flush=True)
        return d
    write_json(d / "config.json", asdict(cfg))

    def cb(t, params, hist):
        if snap_every and t % snap_every == 0 and t > 0:
            np.savez(d / f"snap_{t}.npz", **params)

    t0 = time.time()
    params, hist = train(cfg, verbose=verbose, callback=cb, ckpt_path=str(d / "ckpt.npz"), ckpt_every=5000)
    np.savez(d / "params.npz", **params)
    np.savez(d / "hist.npz", **hist)
    metrics = common.analyse_params(params, cfg, hist)
    metrics["train_seconds_this_session"] = time.time() - t0
    write_json(d / "metrics.json", metrics)
    common.make_run_figures(name, params, cfg, metrics, hist)
    if cfg.dim == 2:
        print(f"[{name}] done: {metrics['n_modules']} modules ({metrics['n_hex_modules']} hexagonal), "
              f"{100 * metrics['frac_grid_analytic']:.0f}% of cells with analytic gridness>0.3, "
              f"actionability error {metrics['actionability']['actionability_rel_error']:.2g}", flush=True)
    else:
        print(f"[{name}] done: {metrics['n_modules']} modules", flush=True)
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", help="name of a run defined in experiments/jobs.py")
    ap.add_argument("--quick", action="store_true", help="1500-step smoke-test version of the job")
    ap.add_argument("--name", help="name for an ad-hoc run (results/runs/<name>)")
    ap.add_argument("--force", action="store_true", help="redo even if metrics.json exists")
    ap.add_argument("--quiet", action="store_true")
    base = TrainConfig()
    for f in fields(TrainConfig):
        v = getattr(base, f.name)
        if isinstance(v, bool):
            ap.add_argument(f"--{f.name}", type=lambda s: s.lower() in ("1", "true", "yes"), default=None)
        else:
            ap.add_argument(f"--{f.name}", type=type(v), default=None)
    a = ap.parse_args()
    over = {f.name: getattr(a, f.name) for f in fields(TrainConfig) if getattr(a, f.name) is not None}
    if a.job:
        job = JOBS[a.job]
        cfgd = quick_cfg(job["cfg"]) if a.quick else dict(job["cfg"])
        cfgd.update(over)
        snap = 0 if a.quick else job["snap_every"]
        run(a.job, TrainConfig(**cfgd), snap, a.force, not a.quiet)
    else:
        if not a.name:
            ap.error("give --job NAME or --name NAME")
        run(a.name, TrainConfig(**over), 0, a.force, not a.quiet)


if __name__ == "__main__":
    main()
