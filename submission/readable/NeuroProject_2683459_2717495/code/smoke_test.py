#!/usr/bin/env python3
"""Quick installation check for the whole submission (about 2 minutes on one CPU core).

    python smoke_test.py               # everything
    python smoke_test.py --skip-jax    # NumPy implementation only (JAX not installed)

Steps (each runs in a fresh process; nothing in the results/ folders is modified):
  1. NumPy implementation: unit tests (finite-difference gradient checks, exact resume, lattice and angle checks)
  2. NumPy implementation: a 300-step training run + analysis in a temporary folder
  3. JAX implementation:   unit tests
  4. JAX implementation:   a 2000-step training run + evaluation in a temporary folder (installation check only)
  5. Cross-check:          shared-input comparison of the JAX equation contract with the NumPy loss/gradient code
Exit status 0 means every step passed. This is NOT a reproduction of the paper's figures; see README.txt for those.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NP = ROOT / "numpy_impl"
JX = ROOT / "jax_impl"
PY = sys.executable


def run(label: str, cmd: list[str], cwd: Path, env_extra: dict | None = None) -> bool:
    env = dict(os.environ, **(env_extra or {}))
    env.setdefault("OMP_NUM_THREADS", "1")
    t0 = time.time()
    print(f"[....] {label}", flush=True)
    proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    dt = time.time() - t0
    ok = proc.returncode == 0
    print(f"[{'PASS' if ok else 'FAIL'}] {label}  ({dt:.0f} s)", flush=True)
    if not ok:
        tail = (proc.stdout + "\n" + proc.stderr).strip().splitlines()[-25:]
        print("\n".join("       " + line for line in tail), flush=True)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-jax", action="store_true", help="skip the JAX implementation and the cross-check")
    args = ap.parse_args()

    for mod in ("numpy", "scipy", "matplotlib"):
        if importlib.util.find_spec(mod) is None:
            print(f"Missing package '{mod}'. Install with:  pip install -r requirements.txt")
            return 2
    have_jax = importlib.util.find_spec("jax") is not None
    if not args.skip_jax and not have_jax:
        print("JAX is not installed, so the JAX steps cannot run. Install with:  pip install -r requirements.txt")
        print("(or run  python smoke_test.py --skip-jax  to check the NumPy implementation only)")
        return 2

    results: list[bool] = []
    with tempfile.TemporaryDirectory(prefix="neuro_smoke_") as tmp:
        tmp = Path(tmp)
        results.append(run("NumPy: unit tests", [PY, "-m", "unittest", "discover", "-s", "tests"], NP))
        results.append(
            run(
                "NumPy: 300-step training run and analysis",
                [PY, "experiments/exp1_grid_emergence.py", "--name", "smoke", "--steps", "300", "--quiet"],
                NP,
                {"ACTIONABLE_RESULTS": str(tmp / "numpy_results"), "MPLCONFIGDIR": str(tmp / "mpl")},
            )
        )
        if not args.skip_jax:
            has_pytest = importlib.util.find_spec("pytest") is not None
            if has_pytest:
                results.append(run("JAX: unit tests", [PY, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"], JX,
                                   {"MPLCONFIGDIR": str(tmp / "mpl")}))
            else:
                print("[SKIP] JAX: unit tests (pytest not installed)")
            results.append(
                run(
                    "JAX: 2000-step training run and evaluation",
                    [PY, "-m", "experiments.run", "--config", "experiments/configs/baseline.json",
                     "--output", str(tmp / "jax_smoke"), "--steps", "2000", "--evaluate"],
                    JX,
                    {"MPLCONFIGDIR": str(tmp / "mpl")},
                )
            )
            results.append(
                run(
                    "Cross-check: JAX equation contract vs NumPy loss/gradient code",
                    [PY, "-m", "comparison.partner_equations", "--output", str(tmp / "audit")],
                    JX,
                    {"MPLCONFIGDIR": str(tmp / "mpl")},
                )
            )

    print()
    if all(results):
        print(f"ALL {len(results)} CHECKS PASSED")
        return 0
    print(f"{len(results) - sum(results)} of {len(results)} CHECKS FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
