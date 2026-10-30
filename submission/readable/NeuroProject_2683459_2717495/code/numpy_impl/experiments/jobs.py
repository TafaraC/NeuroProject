"""The list of training runs behind every number/figure in the report (60 runs).

Every run is one call of experiments/exp1_grid_emergence.py --job <name>.  Jobs are listed in priority order:
the five long "main" runs and the cheap 1-D runs come first, the seed-statistics runs last.
Each entry: name -> dict(group=..., cfg={TrainConfig overrides}, snap_every=<steps between saved snapshots or 0>).
"""
from __future__ import annotations
from collections import OrderedDict

LONG = 150_000          # paper's training length (Table B.3)
MID = 50_000            # shorter runs (ablations, sweeps, sensitivity)
SHORT = 30_000          # 1-D runs

JOBS: "OrderedDict[str, dict]" = OrderedDict()


def add(name, group, steps, snap_every=0, **cfg):
    assert name not in JOBS, name
    JOBS[name] = dict(group=group, cfg=dict(steps=steps, **cfg), snap_every=snap_every)


# ---- main runs: Fig. 2B / Fig. 4 setting (N=64, D=31, sigma=0.2 or 1.0), several seeds
for s in (0, 1, 2):
    add(f"main_sigma0.2_seed{s}", "main", LONG, snap_every=25_000 if s == 0 else 0, sigma=0.2, seed=s)
for s in (0, 1):
    add(f"main_sigma1.0_seed{s}", "main", LONG, snap_every=25_000 if s == 0 else 0, sigma=1.0, seed=s)

# ---- 1-D codes with the simplified loss (eq. 7): lattice of frequencies / harmonics
add("oned_l0", "oned", SHORT, dim=1, N=24, D=11, functional="l0", occupancy="gaussian")
add("oned_l0_uniform", "oned", SHORT, dim=1, N=24, D=11, functional="l0", occupancy="uniform")

# ---- ablations (Appendix I): remove one ingredient at a time (sigma=0.2)
add("abl_no_positivity", "ablation", MID, sigma=0.2, use_positivity=False)
add("abl_no_functional", "ablation", MID, sigma=0.2, functional="none")
add("abl_no_actionability", "ablation", MID, sigma=0.2, D=96)        # 2D+1 = 193 > N: code cannot be actionable

# ---- sweep over the neural length-scale sigma (Section 3.3: number of modules grows as sigma shrinks)
for sg, seeds in ((0.1, (0, 1)), (0.15, (0, 1)), (0.3, (0, 1)), (0.4, (0, 1)), (0.7, (0,))):
    for s in seeds:
        add(f"sweep_sigma{sg}_seed{s}", "sweep_sigma", MID, sigma=sg, seed=s)

# ---- sweep over the number of neurons (Section 4.1 sharpness prediction N ~ (nu/mu)^2 and Fig. B.10)
add("sweepN_N16", "sweep_N", MID, N=16, D=7, sigma=0.5)
add("sweepN_N32", "sweep_N", MID, N=32, D=15, sigma=0.71)
add("sweepN_N100", "sweep_N", MID, N=100, D=49, sigma=1.25)
for n, d, sg in ((25, 12, 0.32), (36, 17, 0.32), (49, 24, 0.39), (64, 31, 0.45)):
    add(f"figB10_N{n}", "sweep_N", MID, N=n, D=d, sigma=sg)

# ---- Fig. 9 of the paper (appendix B.4.2): the sigma sweep at l = 0.2 (sigma = 0.2 is sens_l0.2_sigma0.2_seed0/1 below)
for sg in (0.1, 0.15, 0.3, 0.4, 0.7, 1.0):
    for s in (0, 1):
        add(f"sweepl02_sigma{sg}_seed{s}", "sweep_sigma_l02", MID, sigma=sg, l=0.2, seed=s)

# ---- Fig. 8 of the paper (appendix B.4.1): sweep of l at sigma = 0.4 (l = 0.5 is sweep_sigma0.4_seed0/1), and N = 100 at l = 0.1
for l_ in (1.0, 0.3, 0.2, 0.1):
    add(f"sweepl_l{l_}_sigma0.4", "sweep_l", MID, sigma=0.4, l=l_)
add("sweepl_l0.1_sigma0.4_N100", "sweep_l", MID, sigma=0.4, l=0.1, N=100, D=49)

# ---- sensitivity to choices the paper leaves open
add("sens_kn-4_sigma0.2", "sens", MID, sigma=0.2, k_n=-4.0)
add("sens_kn-4_sigma1.0", "sens", MID, sigma=1.0, k_n=-4.0)
add("sens_D20", "sens", MID, sigma=0.2, D=20)
add("sens_freqstd2", "sens", MID, sigma=0.2, freq_std=2.0)
add("sens_freqstd8", "sens", MID, sigma=0.2, freq_std=8.0)
add("sens_normmean_sigma0.2", "sens", MID, sigma=0.2, norm_mode="mean")
add("sens_normmean_sigma1.0", "sens", MID, sigma=1.0, norm_mode="mean")
for s in (0, 1):
    add(f"sens_l0.2_sigma0.2_seed{s}", "sens", MID, sigma=0.2, l=0.2, seed=s)

# ---- seed statistics at the 50k budget (seed 0 of the baseline is the 'main' run's first 50k steps' sibling)
for s in range(1, 9):
    add(f"base50_sigma0.2_seed{s}", "seeds", MID, sigma=0.2, seed=s)

GROUPS = list(OrderedDict.fromkeys(j["group"] for j in JOBS.values()))
assert len(JOBS) == 60, len(JOBS)


def quick_cfg(cfg: dict) -> dict:
    """Tiny version of a job for smoke tests (--quick): same configuration, 1500 steps."""
    c = dict(cfg)
    c["steps"] = 1500
    c["log_every"] = 100
    return c


if __name__ == "__main__":
    for n, j in JOBS.items():
        print(f"{n:28s} {j['group']:12s} {j['cfg']}")
