"""Run the whole study:  training runs -> analysis -> figures -> LaTeX number macros.

    python experiments/run_all.py                    # everything (60 training runs, ~3.5 CPU-hours in total)
    python experiments/run_all.py --jobs 4           # number of training processes in parallel (default 2)
    python experiments/run_all.py --quick            # 3-minute smoke test of the whole pipeline -> results_quick/
    python experiments/run_all.py --only main oned   # only these groups / job names (see experiments/jobs.py)
    python experiments/run_all.py --reverse          # work through the runs from the END of the list (to share work with a
                                                     # second computer that starts from the beginning)
    python experiments/run_all.py --analyse-only     # no training: re-analyse finished runs, rebuild figures + numbers

Training is resumable: if the process is killed, start the same command again; every run continues from its last
checkpoint (results/runs/<name>/ckpt.npz) and finished runs are skipped.
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from jobs import JOBS, GROUPS                                    # noqa: E402

QUICK_JOBS = ["main_sigma0.2_seed0", "main_sigma1.0_seed0", "abl_no_positivity", "abl_no_functional",
              "oned_l0", "oned_l0_uniform", "sweepN_N16", "sweepN_N32", "sens_D20"]
STAGES = ["exp2_1d_lattice.py", "exp3_sharpness.py", "exp4_module_angle.py", "exp5_ablations_sigma.py",
          "exp6_sensitivity.py", "make_figures.py", "make_numbers.py"]


def stamp():
    return time.strftime("%H:%M:%S")


def run_training(names, n_parallel, quick, results_dir):
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               ACTIONABLE_RESULTS=str(results_dir))
    runs = Path(results_dir) / "runs"
    todo = [n for n in names if not (runs / n / "metrics.json").exists()]
    print(f"[{stamp()}] {len(todo)} of {len(names)} training runs still to do ({n_parallel} in parallel)", flush=True)

    def one(name):
        d = runs / name
        d.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(HERE / "exp1_grid_emergence.py"), "--job", name] + (["--quick"] if quick else [])
        t0 = time.time()
        with open(d / "train.log", "a") as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
        print(f"[{stamp()}] {name}: {'ok' if rc == 0 else 'FAILED rc=%d (see %s)' % (rc, d / 'train.log')} "
              f"({time.time() - t0:.0f}s)", flush=True)
        return rc

    with ThreadPoolExecutor(max_workers=n_parallel) as ex:
        rcs = list(ex.map(one, todo))
    return sum(r != 0 for r in rcs)


def run_stage(script, results_dir):
    env = dict(os.environ, ACTIONABLE_RESULTS=str(results_dir))
    p = HERE / script
    if not p.exists():
        print(f"[{stamp()}] (skipping missing {script})", flush=True)
        return 0
    t0 = time.time()
    rc = subprocess.run([sys.executable, str(p)], env=env).returncode
    print(f"[{stamp()}] {script}: {'ok' if rc == 0 else 'FAILED rc=%d' % rc} ({time.time() - t0:.0f}s)", flush=True)
    return rc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=min(2, os.cpu_count() or 1), help="parallel training processes")
    ap.add_argument("--quick", action="store_true", help="smoke test: few 1500-step runs, results in results_quick/")
    ap.add_argument("--only", nargs="*", help="job names or group names (%s)" % ", ".join(GROUPS))
    ap.add_argument("--analyse-only", action="store_true", help="skip training")
    ap.add_argument("--reverse", action="store_true", help="process the runs in reverse order")
    a = ap.parse_args()

    results_dir = ROOT / ("results_quick" if a.quick else "results")
    names = list(JOBS)
    if a.quick:
        names = QUICK_JOBS
    if a.only:
        names = [n for n in JOBS if n in a.only or JOBS[n]["group"] in a.only]
        if not names:
            sys.exit(f"nothing matches --only {a.only}")
    if a.reverse:
        names = names[::-1]
    failed = 0
    t0 = time.time()
    if not a.analyse_only:
        failed += run_training(names, a.jobs, a.quick, results_dir)
    else:
        failed += run_stage("reanalyse.py", results_dir)
    if a.only and not a.analyse_only:
        print(f"[{stamp()}] --only given: skipping the analysis stages (run again without --only, or with "
              f"--analyse-only, when all training runs exist)")
    else:
        for s in STAGES:
            failed += run_stage(s, results_dir)
    print(f"[{stamp()}] finished in {(time.time() - t0) / 60:.1f} min; {failed} step(s) failed. "
          f"Results: {results_dir}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
