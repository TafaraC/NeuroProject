"""Experiment 4 - Section 4.2 / Appendix K: modules should be rotated by small angles (~4 deg) relative to each other.

(A) Re-computation of the paper's prediction (eqs. 99-102, see src/actionable/module_angle.py): overlap of the
    smoothed angular frequency density of two identical modules as a function of their relative rotation, its
    minimum, the three-module optimum, and how the minimum depends on the two free parameters (mu, nu).
(B) Measurement on our trained networks: relative orientation of the lattices of every pair of hexagonal modules
    in the same network, folded to [0, 30] deg (60 deg lattice symmetry + mirror), against the null of independent
    uniform orientations (mean 15 deg; P[offset < 8 deg] = 0.27).

Output: results/tables/exp4_module_angle.json, results/figures/exp4_module_angle.png
"""
from __future__ import annotations
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common                                                    # noqa: E402
from common import RUNS, FIGS, TABLES, read_json, write_json     # noqa: E402
from actionable import module_angle as ma                        # noqa: E402


def fold60(d):
    d = np.mod(d, 60.0)
    return np.minimum(d, 60.0 - d)


def prediction():
    mus = np.geomspace(0.03, 0.4, 12)
    nus = np.geomspace(0.1, 0.6, 12)
    grid = np.zeros((len(nus), len(mus)))
    for i, nu in enumerate(nus):
        for j, mu in enumerate(mus):
            grid[i, j] = ma.optimal_offset(mu, nu, n_psi=720)[0]
    # reference parameters: the scanned pair whose optimum is closest to the paper's 4 degrees (refined at 0.25 deg resolution)
    i0, j0 = np.unravel_index(np.argmin(np.abs(grid - 4.0)), grid.shape)
    mu0, nu0 = float(mus[j0]), float(nus[i0])
    d0 = ma.optimal_offset(mu0, nu0, n_psi=1440)[0]
    ang, O = ma.overlap_curve(ma.angular_density(mu0, nu0, 1440))
    (a3, b3), _ = ma.three_module_offsets(mu0, nu0, n_psi=720)
    interior = grid < 29.0                                   # optimum is a genuine interior minimum, not the 30-deg edge
    band = (grid >= 3.0) & (grid <= 8.0)
    return dict(mus=mus.tolist(), nus=nus.tolist(), grid=grid.tolist(),
                ref=dict(mu=float(mu0), nu=float(nu0), optimum_deg=float(d0), three_module_offsets=[float(a3), float(b3)]),
                scan_min=float(grid.min()), scan_max=float(grid.max()),
                frac_interior=float(interior.mean()), frac_in_3_to_8=float(band.mean()),
                frac_in_3_to_8_of_interior=float(band.sum() / max(interior.sum(), 1)),
                angles=ang.tolist(), overlap=(O / O.max()).tolist())


def measurement(names):
    pairs, per_run = [], {}
    for n in names:
        m = read_json(RUNS / n / "metrics.json")
        o = m.get("hex_orientations_deg") or []
        if len(o) < 2:
            continue
        ds = [float(fold60(a - b)) for a, b in itertools.combinations(o, 2)]
        per_run[n] = dict(orientations=o, offsets=ds, spacings=m["hex_spacings"])
        pairs += ds
    pairs = np.array(pairs)
    out = dict(per_run=per_run, n_pairs=int(len(pairs)), n_runs=len(per_run))
    if len(pairs):
        out.update(median=float(np.median(pairs)), mean=float(pairs.mean()), null_mean=15.0,
                   frac_below_8=float((pairs < 8).mean()), null_frac_below_8=8 / 30,
                   frac_3_to_8=float(((pairs >= 3) & (pairs <= 8)).mean()),
                   frac_below_1=float((pairs < 1.0).mean()), pairs=pairs.tolist())
    return out


def main():
    common.ensure_dirs()
    pred = prediction()
    names = common.finished_runs()
    meas = measurement(names)
    write_json(TABLES / "exp4_module_angle.json", dict(prediction=pred, measurement=meas))
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(9.2, 2.6))
    ax[0].plot(pred["angles"], pred["overlap"], "k-", lw=1)
    ax[0].axvline(pred["ref"]["optimum_deg"], color="r", lw=0.8)
    ax[0].axvline(-pred["ref"]["optimum_deg"], color="r", lw=0.8)
    ax[0].set_xlim(-65, 65); ax[0].set_xlabel("relative rotation (deg)"); ax[0].set_ylabel("overlap (normalised)")
    ax[0].set_title(f"mu={pred['ref']['mu']:.3f}, nu={pred['ref']['nu']:.2f}: optimum {pred['ref']['optimum_deg']:.1f} deg", fontsize=8)
    im = ax[1].imshow(np.array(pred["grid"]), origin="lower", cmap="magma", aspect="auto", vmin=0, vmax=30,
                      extent=[0, len(pred["mus"]), 0, len(pred["nus"])])
    ax[1].set_xticks(np.arange(0, len(pred["mus"]), 3) + 0.5); ax[1].set_xticklabels([f"{m:.2f}" for m in pred["mus"][::3]], fontsize=6)
    ax[1].set_yticks(np.arange(0, len(pred["nus"]), 3) + 0.5); ax[1].set_yticklabels([f"{m:.2f}" for m in pred["nus"][::3]], fontsize=6)
    ax[1].set_xlabel(r"$\mu$ (peak width)"); ax[1].set_ylabel(r"$\nu$ (lattice length)")
    plt.colorbar(im, ax=ax[1]).set_label("optimal offset (deg)", fontsize=7)
    if meas["n_pairs"]:
        ax[2].hist(meas["pairs"], bins=np.linspace(0, 30, 11), color="gray", density=True)
        ax[2].axhline(1 / 30, color="r", lw=1, label="uniform null")
        ax[2].axvspan(3, 8, color="g", alpha=0.15, label="paper: 3-8 deg")
        ax[2].legend(fontsize=6)
    ax[2].set_xlabel("offset between hexagonal modules (deg)"); ax[2].set_title(f"{meas['n_pairs']} pairs from {meas['n_runs']} runs", fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS / "exp4_module_angle.png", dpi=170); plt.close(fig)
    print(f"exp4: reference (mu, nu) = ({pred['ref']['mu']:.3f}, {pred['ref']['nu']:.2f}) -> {pred['ref']['optimum_deg']:.1f} deg, "
          f"three modules {pred['ref']['three_module_offsets']}; scan range {pred['scan_min']:.1f}-{pred['scan_max']:.1f} deg, "
          f"{100 * pred['frac_in_3_to_8']:.0f}% of scan within 3-8 deg")
    if meas["n_pairs"]:
        print(f"      measured: {meas['n_pairs']} module pairs in {meas['n_runs']} runs; median offset {meas['median']:.1f} deg "
              f"(null 15); {100 * meas['frac_below_8']:.0f}% below 8 deg (null 27%)")


if __name__ == "__main__":
    main()
