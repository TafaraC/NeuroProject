Dorrell et al. (2023) independent reproduction

Use Python 3.14 (64-bit). Run all commands from the repository root in PowerShell.

SETUP
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip check

VALIDATE
.\.venv\Scripts\python.exe -m pytest tests -q

TRAIN AND EVALUATE
.\.venv\Scripts\python.exe -m experiments.run --config experiments/configs/baseline.json --output results/data/baseline_seed0 --evaluate
.\.venv\Scripts\python.exe -m experiments.run --config experiments/configs/fine_resolution.json --output results/data/fine_seed0 --evaluate

COMPARE
.\.venv\Scripts\python.exe -m experiments.compare results/data/baseline_seed0/evaluation_150000 results/data/fine_seed0/evaluation_150000 --output results/figures/comparison_seed0

Existing output directories are protected against overwrite. Use a new directory
for each new run. To continue the same run after an interruption:
.\.venv\Scripts\python.exe -m experiments.run --resume results/data/baseline_seed0/checkpoint.npz --output results/data/baseline_seed0 --evaluate

The default is 150000 steps per condition, N=64, sigma=0.4, and l=0.5 versus 0.1.
Add --steps 2000 with a new output directory for a short installation check only.
Additional paired seeds: add --seed 1 and use new directories for both conditions.

See README.md and references/implementation_notes.md for model details,
paper/code discrepancies, analysis conventions, and limits of interpretation.
The implementation uses a specified norm tolerance that differs from the public
code. Check held-out positivity and norm losses before declaring success.
One training seed per condition is not a robustness study.

Raw run folders are excluded from Git; retain them locally or include them
separately when sharing results. Compact comparison figures/CSV can be committed.

COMPLETED SEED-0 ATTEMPT
See results/reference_runs/README.md and results/figures/comparison_seed0.
Both 150000-step runs completed, but did not cleanly recover the intended
hexagonal baseline; the baseline also missed its held-out norm target.
Versioned checkpoint snapshots and evaluation figures are included under
results/reference_runs. Large map arrays remain in ignored results/data folders.
