#!/usr/bin/env python3
"""Turn result files into the LaTeX macros used by main.tex, so that no number in the review is typed by hand.

    python make_review_numbers.py                 # copy the NumPy macros/tables, write numbers_extra.tex
    python make_review_numbers.py --regenerate    # first re-run numpy_impl/experiments/make_numbers.py

Reads only files that are part of this submission:
  numpy_impl/results/{runs,tables}/...   (NumPy implementation; 60 training runs)
  jax_impl/results/...                   (JAX implementation; reference runs, diagnostic study, controls, audit)
Writes (next to this script):
  numbers_numpy.tex            copy of numpy_impl/review/tables/numbers.tex (written by numpy_impl/experiments/make_numbers.py)
  tables/*.tex                 copies of the NumPy LaTeX tables
  numbers_extra.tex            additional macros computed here (prefixes: x = NumPy side, j = JAX side, audit = cross-check)
Macro names contain letters only (a LaTeX rule).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
NP = ROOT / "numpy_impl"
JX = ROOT / "jax_impl"
facts: dict[str, str] = {}


def put(name: str, value) -> None:
    assert re.fullmatch(r"[A-Za-z]+", name), f"bad macro name {name}"
    facts[name] = str(value)


def jload(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def pct(x: float) -> str:
    return f"{100 * x:.0f}\\%"


def sci(x: float, digits: int = 1) -> str:
    """2.77e-13 -> $2.8\\times10^{-13}$"""
    import math

    if x == 0:
        return "$0$"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10**e
    return f"${m:.{digits}f}\\times10^{{{e}}}$"


# ------------------------------------------------------------------------------------------------ NumPy side
def numpy_side(regenerate: bool) -> None:
    if regenerate:
        subprocess.run([sys.executable, str(NP / "experiments" / "make_numbers.py")], check=True, cwd=NP)
    src = NP / "review" / "tables"
    (HERE / "tables").mkdir(exist_ok=True)
    def copy_lf(a: Path, b: Path) -> None:                       # the NumPy files may have been written on Windows (CRLF)
        b.write_text(a.read_text(encoding="utf-8").replace("\r\n", "\n"), encoding="utf-8", newline="\n")

    copy_lf(src / "numbers.tex", HERE / "numbers_numpy.tex")
    for t in ("sigma_table.tex", "ablation_table.tex", "sensitivity.tex"):
        copy_lf(src / t, HERE / "tables" / t)

    runs = NP / "results" / "runs"

    def M(name: str, fn: str = "metrics.json"):
        f = runs / name / fn
        return jload(f) if f.exists() else None

    # norm penalty at the end of the five main runs (eq. 25 loss; literal target k_n = +4)
    mains = [M(f"main_sigma{sg}_seed{s}") for sg, seeds in (("0.2", (0, 1, 2)), ("1.0", (0, 1))) for s in seeds]
    mains = [m for m in mains if m]
    lb = [m["final_Lb"] for m in mains]
    put("xLbMainMin", f"{min(lb):.1f}")
    put("xLbMainMax", f"{max(lb):.1f}")
    put("xLamMainAll", "all" if all(m["final_lam_n"] < 1e-6 for m in mains) else "some")
    put("xMainLowTotalModules", sum(m["n_modules"] for m in mains[:3]))
    put("xMainLowTotalHex", sum(m["n_hex_modules"] for m in mains[:3]))
    put("xMainLowGridMax", pct(max(m["frac_grid_analytic"] for m in mains[:3])))

    # one matched pair: literal (inactive) vs tight (active) norm target, sigma = 1.0, seed 0, 50k steps
    loose1 = M("main_sigma1.0_seed0", "metrics_snap50000.json")
    tight1 = M("sens_kn-4_sigma1.0")
    tight2 = M("sens_kn-4_sigma0.2")
    if loose1 and tight1:
        put("xLooseOneModules", loose1["n_modules"]); put("xLooseOneHex", loose1["n_hex_modules"])
        put("xLooseOneGrid", pct(loose1["frac_grid_analytic"]))
        put("xTightOneModules", tight1["n_modules"]); put("xTightOneHex", tight1["n_hex_modules"])
        put("xTightOneGrid", pct(tight1["frac_grid_analytic"]))
        put("xTightOneLb", f"{tight1['final_Lb']:.3f}"); put("xTightOneLam", f"{tight1['final_lam_n']:.1f}")
    if tight2:
        put("xTightLowModules", tight2["n_modules"]); put("xTightLowHex", tight2["n_hex_modules"])
        put("xTightLowGrid", pct(tight2["frac_grid_analytic"]))
        put("xTightLowLb", f"{tight2['final_Lb']:.3f}"); put("xTightLowLam", f"{tight2['final_lam_n']:.2f}")

    # Figure 8 claims at sigma = 0.4 (50k steps): two seeds at l = 0.5; l = 0.1 with 64 and 100 neurons
    for tag, name in (("A", "sweep_sigma0.4_seed0"), ("B", "sweep_sigma0.4_seed1")):
        m = M(name)
        if m:
            put(f"xHalfModules{tag}", m["n_modules"]); put(f"xHalfHex{tag}", m["n_hex_modules"])
    for tag, name in (("N", "sweepl_l0.1_sigma0.4"), ("H", "sweepl_l0.1_sigma0.4_N100")):
        m = M(name)
        if m:
            put(f"xLtenModules{tag}", m["n_modules"]); put(f"xLtenHex{tag}", m["n_hex_modules"])

    # lattice spacing of the single module in the two sigma = 1.0 runs of the montage figure
    for tag, name in (("Loose", "main_sigma1.0_seed0"), ("Tight", "sens_kn-4_sigma1.0")):
        m = M(name)
        put(f"xSpacing{tag}", f"{m['modules'][0]['ac_spacing']:.1f}")
    # literal-target 50k-step baseline runs at sigma = 0.2: norm loss and multiplier at the end of training
    b50 = [M(f"base50_sigma0.2_seed{s}") for s in range(1, 9)]
    b50 = [m for m in b50 if m]
    put("xBaseLbLo", f"{min(m['final_Lb'] for m in b50):.1f}")
    put("xBaseLbHi", f"{max(m['final_Lb'] for m in b50):.1f}")
    put("xBaseLamAtFloor", sum(m["final_lam_n"] < 1e-6 for m in b50))
    put("xBaseLamRuns", len(b50))

    # computing cost
    secs = [m["train_seconds_this_session"] for m in (M(d.name) for d in sorted(runs.iterdir()) if d.is_dir())
            if m and "train_seconds_this_session" in m]
    if secs:
        put("xCpuHours", f"{sum(secs) / 3600:.1f}")
        put("xRunsTimed", len(secs))


# ------------------------------------------------------------------------------------------------ JAX side
def jax_side() -> None:
    res = JX / "results"
    for tag, name in (("Half", "baseline_seed0"), ("Tenth", "fine_seed0")):
        s = jload(res / "reference_runs" / name / "evaluation" / "summary.json")
        put(f"jRef{tag}Score", f"{s['grid_score_median']:.3f}")
        put(f"jRef{tag}Norm", f"{s['norm_loss']:.3f}")
        put(f"jRef{tag}Neg", f"{100 * s['negative_fraction_below_minus_1e_minus6']:.1f}\\%")
        put(f"jRef{tag}NormMet", "met" if s["norm_target_met"] else "missed")
    rows = list(csv.DictReader((res / "diagnostic_study" / "report" / "runs.csv").read_text(encoding="utf-8-sig").splitlines()))
    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(r["condition"], []).append(r)
    for tag, cond in (("Base", "baseline"), ("Var", "sigma_variance"), ("Tab", "norm_table")):
        sc = [float(r["grid_score_median"]) for r in by[cond]]
        nl = [float(r["norm_loss"]) for r in by[cond]]
        put(f"jDiag{tag}ScoreLo", f"{min(sc):.3f}"); put(f"jDiag{tag}ScoreHi", f"{max(sc):.3f}")
        put(f"jDiag{tag}NormLo", f"{min(nl):.3g}"); put(f"jDiag{tag}NormHi", f"{max(nl):.3g}")
        put(f"jDiag{tag}Seeds", len(sc))
    put("jDiagRuns", len(rows))
    put("jDiagBothMet", sum(r["common_targets_met"] == "True" for r in rows))
    steps = {int(r["steps"]) for r in rows}
    put("jDiagSteps", f"{min(steps):,}".replace(",", "\\,"))

    c = jload(res / "constraint_control" / "summary.json")
    for tag, m in zip(("Fifty", "OneFifty", "SixHundred"), c["measurements"]):
        put(f"jCtl{tag}", f"{m['mean_norm_loss']:.3g}")
        put(f"jCtl{tag}Frac", pct(m["fraction_of_batches_at_or_below_target"]))
    put("jCtlTarget", f"{c['target_under_test']:.2f}")
    put("jCtlRel", sci(c["maximum_analytic_room_energy_relative_error"], 0))

    a = jload(res / "partner_comparison" / "comparison.json")
    e = a["absolute_errors"]
    put("auditLoss", sci(max(e["functional"], e["positivity"], e["norm"]), 1))
    put("auditGrad", sci(max(e["weight_directional_derivative"], e["frequency_directional_derivative"]), 1))
    put("auditCommit", a["partner_revision"][:7])
    put("auditPassed", "passes" if a["passed_matched_equations"] else "FAILS")

    # wall-clock time of the two reference runs (sum of recorded block times; they ran concurrently)
    mins = []
    for name in ("baseline_seed0", "fine_seed0"):
        h = res / "reference_runs" / name / "history.jsonl"
        mins.append(sum(json.loads(line)["block_seconds"] for line in h.read_text(encoding="utf-8").splitlines() if line.strip()) / 60)
    put("jRefMinutes", f"{sum(mins) / len(mins):.0f}")

    # wall-clock time of the nine diagnostic runs (sum of recorded block times)
    tot = 0.0
    for h in sorted((res / "diagnostic_study" / "runs").glob("*/history.jsonl")):
        tot += sum(json.loads(line)["block_seconds"] for line in h.read_text(encoding="utf-8").splitlines() if line.strip())
    if tot:
        put("jDiagMinutes", f"{tot / 60:.0f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regenerate", action="store_true", help="re-run numpy_impl/experiments/make_numbers.py first")
    a = ap.parse_args()
    numpy_side(a.regenerate)
    jax_side()
    out = HERE / "numbers_extra.tex"
    with out.open("w", encoding="utf-8") as fh:
        fh.write("% generated by make_review_numbers.py - do not edit\n")
        for k in sorted(facts):
            fh.write(f"\\newcommand{{\\{k}}}{{{facts[k]}}}\n")
    print(f"make_review_numbers: wrote {len(facts)} macros to {out.name} (+ numbers_numpy.tex, tables/)")


if __name__ == "__main__":
    main()
