"""Write every number quoted in the review as a LaTeX macro (review/tables/numbers.tex) and the LaTeX tables
(review/tables/sigma_table.tex, review/tables/sensitivity.tex, review/tables/ablation_table.tex).

The review text only ever uses these macros, so text and results cannot drift apart: run
    python experiments/make_numbers.py
after any change in the results, then rebuild the PDF.  Macro names may contain letters only (LaTeX rule).
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common                                                    # noqa: E402
from common import ROOT, RUNS, TABLES, read_json                  # noqa: E402

OUT = ROOT / "review" / "tables" if common.RESULTS == ROOT / "results" else common.RESULTS / "review_tables"
facts = {}


def put(name, value):
    assert re.fullmatch(r"[A-Za-z]+", name), f"bad macro name {name}"
    facts[name] = str(value)


def f0(x):
    return f"{x:.0f}"


def f1(x):
    return f"{x:.1f}"


def f2(x):
    return f"{x:.2f}"


def pct(x):
    return f"{100 * x:.0f}\\%"


def sci(x):
    if x == 0:
        return "0"
    e = int(np.floor(np.log10(abs(x))))
    return f"{x / 10 ** e:.0f}\\times10^{{{e}}}"


def safe(fn):
    def w():
        try:
            fn()
        except Exception as e:                                   # a missing run must not stop the other numbers
            print(f"make_numbers: {fn.__name__} skipped ({e!r})")
    return w


def M(name):
    f = RUNS / name / "metrics.json"
    return read_json(f) if f.exists() else None


LET = "ABCDEFGH"


@safe
def main_runs():
    for tag, sg, seeds in (("Low", "0.2", (0, 1, 2)), ("High", "1.0", (0, 1))):
        ms = []
        for i, s in enumerate(seeds):
            m = M(f"main_sigma{sg}_seed{s}")
            if not m:
                continue
            ms.append(m)
            put(f"main{tag}Modules{LET[i]}", m["n_modules"])
            put(f"main{tag}Hex{LET[i]}", m["n_hex_modules"])
            put(f"main{tag}Sizes{LET[i]}", "/".join(str(mod["size"]) for mod in sorted(m["modules"], key=lambda t: -t["size"])))
            put(f"main{tag}Grid{LET[i]}", pct(m["frac_grid_analytic"]))
        if not ms:
            continue
        put(f"main{tag}ModulesMean", f1(np.mean([m["n_modules"] for m in ms])))
        put(f"main{tag}ModulesMin", min(m["n_modules"] for m in ms))
        put(f"main{tag}ModulesMax", max(m["n_modules"] for m in ms))
        put(f"main{tag}HexMean", f1(np.mean([m["n_hex_modules"] for m in ms])))
        put(f"main{tag}GridFrac", pct(np.mean([m["frac_grid_analytic"] for m in ms])))
        put(f"main{tag}GridFracWin", pct(np.mean([m["frac_grid_windowed"] for m in ms])))
        put(f"main{tag}InModules", pct(np.mean([m["frac_in_modules"] for m in ms])))
        put(f"main{tag}NegMass", f"{np.mean([m['neg_mass'] for m in ms]):.3f}")
        put(f"main{tag}NRuns", len(ms))
        if tag == "Low":
            rat = [r for m in ms for r in m["hex_spacing_ratios"]]
            sp = [s for m in ms for s in m["hex_spacings"]]
            if rat:
                put("mainRatioMedian", f2(np.median(rat))); put("mainRatioMin", f2(min(rat))); put("mainRatioMax", f2(max(rat)))
                put("mainRatioN", len(rat))
            if sp:
                put("mainSpacingMin", f2(min(sp))); put("mainSpacingMax", f2(max(sp)))
    allm = [M(n) for n in ("main_sigma0.2_seed0", "main_sigma0.2_seed1", "main_sigma0.2_seed2",
                           "main_sigma1.0_seed0", "main_sigma1.0_seed1")]
    allm = [m for m in allm if m]
    if allm:
        put("actErrMax", "$" + sci(max(m["actionability"]["actionability_rel_error"] for m in allm)) + "$")
        put("actRankMin", f"{min(m['actionability']['rank_ratio'] for m in allm):.0e}".replace("e-0", "e-").replace("e-", "\\times10^{-") + "}")
        put("lamNFinal", "$10^{-8}$" if all(m.get("final_lam_n", 1) < 1e-6 for m in allm) else f"{np.mean([m['final_lam_n'] for m in allm]):.2g}")
        put("lbFinalMean", f2(np.mean([m["final_Lb"] for m in allm])))
        put("lpFinalMean", f"{np.mean([m['final_Lp'] for m in allm]):.1g}")
    secs = []
    for n in ("main_sigma0.2_seed0", "main_sigma0.2_seed1", "main_sigma0.2_seed2"):
        f = RUNS / n / "train.log"
        if f.exists() and "resumed" not in f.read_text():
            m = M(n)
            if m:
                secs.append(m["train_seconds_this_session"])
    if secs:
        put("trainMinLong", f0(np.median(secs) / 60))


@safe
def ablations():
    r = read_json(TABLES / "exp5_ablations_sigma.json")["ablations"]
    b = r["baseline"]
    put("baseModules", f1(b["n_modules"]["mean"])); put("baseModulesSd", f1(b["n_modules"]["sd"]))
    put("baseHex", f1(b["n_hex_modules"]["mean"])); put("baseGrid", pct(b["frac_grid_analytic"]["mean"]))
    put("baseNeff", f1(b["neff"]["mean"])); put("baseConst", f2(b["const_share"]["mean"]))
    put("baseNeg", f"{b['neg_mass']['mean']:.3f}"); put("baseSeeds", b["n_runs"])
    for tag, k in (("NoPos", "no_positivity"), ("NoFun", "no_functional"), ("NoAct", "no_actionability")):
        if k in r:
            v = r[k]
            put(f"abl{tag}Modules", f0(v["n_modules"]["mean"])); put(f"abl{tag}Hex", f0(v["n_hex_modules"]["mean"]))
            put(f"abl{tag}Grid", pct(v["frac_grid_analytic"]["mean"])); put(f"abl{tag}Neff", f1(v["neff"]["mean"]))
            put(f"abl{tag}Const", f2(v["const_share"]["mean"])); put(f"abl{tag}Neg", f"{v['neg_mass']['mean']:.3f}")
            put(f"abl{tag}Min", f2(v["min_rate_mean"]["mean"]))
    rows = [("baseline (9 seeds)", b)] + [(lab, r[k]) for lab, k in (("no non-negativity", "no_positivity"),
                                                                       ("no functional term", "no_functional"),
                                                                       ("no actionability ($D{=}96$)", "no_actionability")) if k in r]
    lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
             r"& modules & hex. & $>$0.3 & $n_\mathrm{eff}$ & $a_0^2$ share & neg. mass \\", r"\midrule"]
    for lab, v in rows:
        lines.append(f"{lab} & {v['n_modules']['mean']:.1f} & {v['n_hex_modules']['mean']:.1f} & {pct(v['frac_grid_analytic']['mean'])} & "
                     f"{v['neff']['mean']:.1f} & {v['const_share']['mean']:.2f} & {v['neg_mass']['mean']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "ablation_table.tex").write_text("\n".join(lines) + "\n")


@safe
def sigma_tables():
    t = read_json(TABLES / "exp5_ablations_sigma.json")["sigma_table"]
    sigs = sorted({float(s) for lk in t for s in t[lk]})
    lines = [r"\begin{tabular}{l" + "c" * len(sigs) + "}", r"\toprule",
             r"$\sigma$ & " + " & ".join(f"{s:g}" for s in sigs) + r" \\", r"\midrule"]
    for lk in ("l=0.5", "l=0.2"):
        if lk not in t:
            continue
        cells, cellsh = [], []
        for s in sigs:
            v = t[lk].get(str(s)) or t[lk].get(f"{s}")
            if v:
                cells.append(f"{v['n_modules'][0]:.1f}" + (f"$\\pm${v['n_modules'][1]:.1f}" if v["n"] > 1 else ""))
                cellsh.append(f"{v['n_hex'][0]:.1f}")
            else:
                cells.append("--"); cellsh.append("--")
        lines.append(f"modules, ${lk.replace('=', '{=}')}$ & " + " & ".join(cells) + r" \\")
        lines.append(f"~~hexagonal & " + " & ".join(cellsh) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "sigma_table.tex").write_text("\n".join(lines) + "\n")
    # headline facts
    lo, hi = t.get("l=0.5", {}), None
    ks = sorted(lo, key=float)
    if ks:
        put("sigmaSmallModules", f1(lo[ks[0]]["n_modules"][0])); put("sigmaSmall", ks[0])
        put("sigmaLargeModules", f1(lo[ks[-1]]["n_modules"][0])); put("sigmaLarge", ks[-1])
    ks2 = sorted(t.get("l=0.2", {}), key=float)
    if ks2:
        put("sigmaTwoSmallModules", f1(t["l=0.2"][ks2[0]]["n_modules"][0])); put("sigmaTwoLargeModules", f1(t["l=0.2"][ks2[-1]]["n_modules"][0]))
        put("sigmaTwoLargeHex", f1(t["l=0.2"][ks2[-1]]["n_hex"][0])); put("sigmaTwoLarge", ks2[-1])


@safe
def sensitivity():
    r = read_json(TABLES / "exp6_sensitivity.json")
    lines = [r"\begin{tabular}{lccccc}", r"\toprule",
             r"change (one at a time) & $\sigma$ & modules & hex. & $>$0.3 & vs.\ baseline \\", r"\midrule"]
    for bk in ("sigma0.2", "sigma1.0"):
        v = r["baseline"].get(bk)
        if v:
            lines.append(f"baseline ({v['n_runs']} run{'s' if v['n_runs'] > 1 else ''}) & {bk[5:]} & {v['n_modules']['mean']:.1f} & "
                         f"{v['n_hex_modules']['mean']:.1f} & {pct(v['frac_grid_analytic']['mean'])} & -- \\\\")
    for row in r["rows"]:
        b = r["baseline"].get(row["baseline"])
        d = ""
        if b:
            diff = row["n_modules"]["mean"] - b["n_modules"]["mean"]
            d = f"{diff:+.1f} modules"
        lines.append(f"{row['label']} & {row['baseline'][5:]} & {row['n_modules']['mean']:.1f} & {row['n_hex_modules']['mean']:.1f} & "
                     f"{pct(row['frac_grid_analytic']['mean'])} & {d} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "sensitivity.tex").write_text("\n".join(lines) + "\n")


@safe
def one_d():
    r = read_json(TABLES / "exp2_1d.json")
    a = r.get("oned_l0")
    if a:
        put("oneDFund", f2(a["fundamental"])); put("oneDDev", f"{a['power_weighted_deviation_from_integer']:.2f}")
        put("oneDNull", f"{a['null_deviation_median']:.2f}"); put("oneDShare", pct(a["share_power_within_0p1_of_integer"]))
        put("oneDMeanRms", f2(a["mean_over_rms_mean"])); put("oneDNeg", f"{100 * a['frac_negative']:.1f}\\%")
        put("oneDNeff", f1(a["n_effective_freqs"]))
    b = r.get("oned_l0_uniform")
    if b:
        put("oneDUniDev", f"{b['power_weighted_deviation_from_integer']:.2f}"); put("oneDUniMeanRms", f2(b["mean_over_rms_mean"]))
        put("oneDUniShare", pct(b["share_power_within_0p1_of_integer"]))


@safe
def sharpness():
    r = read_json(TABLES / "exp3_sharpness.json")
    f = r.get("fit")
    if f:
        put("sharpSlope", f2(f["slope"])); put("sharpCILo", f2(f["slope_ci95"][0])); put("sharpCIHi", f2(f["slope_ci95"][1]))
        put("sharpSpearman", f2(f["spearman"])); put("sharpNMod", f["n_modules"]); put("sharpNRuns", f["n_runs"])
        put("sharpR", f2(f["r"]))


@safe
def angles():
    r = read_json(TABLES / "exp4_module_angle.json")
    p, m = r["prediction"], r["measurement"]
    put("angleRef", f1(p["ref"]["optimum_deg"])); put("angleRefMu", f"{p['ref']['mu']:.3f}"); put("angleRefNu", f2(p["ref"]["nu"]))
    put("angleThreeA", f1(p["ref"]["three_module_offsets"][0])); put("angleThreeB", f1(p["ref"]["three_module_offsets"][1]))
    put("angleScanMin", f1(p["scan_min"])); put("angleScanMax", f0(p["scan_max"]))
    put("angleBandFrac", pct(p["frac_in_3_to_8"])); put("angleInterior", pct(p["frac_interior"]))
    put("angleBandOfInterior", pct(p["frac_in_3_to_8_of_interior"]))
    if m.get("n_pairs"):
        put("angleMeasPairs", m["n_pairs"]); put("angleMeasRuns", m["n_runs"]); put("angleMeasMedian", f1(m["median"]))
        put("angleMeasLow", pct(m["frac_below_8"])); put("angleMeasBand", pct(m["frac_3_to_8"]))


@safe
def misc():
    t = (ROOT / "tests" / "test_core.py").read_text()
    put("nTests", len(re.findall(r"^\s+def test_", t, flags=re.M)))
    put("nRunsFinished", len(common.finished_runs()))
    from jobs import JOBS
    put("nJobs", len(JOBS))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (main_runs, ablations, sigma_tables, sensitivity, one_d, sharpness, angles, misc):
        f()
    with open(OUT / "numbers.tex", "w") as fh:
        fh.write("% generated by experiments/make_numbers.py - do not edit\n")
        for k in sorted(facts):
            fh.write(f"\\newcommand{{\\{k}}}{{{facts[k]}}}\n")
    print(f"make_numbers: wrote {len(facts)} macros to {OUT / 'numbers.tex'}")


if __name__ == "__main__":
    main()
