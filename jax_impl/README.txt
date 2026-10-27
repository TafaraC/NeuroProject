JAX implementation of Dorrell et al. (2023): written by Tafara Chibanda, with OpenAI Codex.
(Part of the merged submission: see ../README.txt for the overview. The NumPy implementation is in ../numpy_impl.)

Run every command from THIS folder (jax_impl). Commands are shown for any OS ("python" = the Python of your
environment; on Windows PowerShell use the same commands, e.g. .\.venv\Scripts\python.exe instead of python).
Developed and verified on 64-bit Python 3.14 on Windows; the test-suite also passes on Python 3.12 (Linux).

SETUP (one of)
  pip install -r ../requirements.txt          # loose requirements for the whole submission (recommended)
  pip install -r requirements-dev.txt         # exact pins used during development (Windows x64, Python 3.14)

VALIDATE  (about 30 s)
  python -m pytest tests -q

TRAIN AND EVALUATE  (about 10 min per 150000-step run on one CPU)
  python -m experiments.run --config experiments/configs/baseline.json --output results/data/baseline_seed0 --evaluate
  python -m experiments.run --config experiments/configs/fine_resolution.json --output results/data/fine_seed0 --evaluate

COMPARE
  python -m experiments.compare results/data/baseline_seed0/evaluation_150000 results/data/fine_seed0/evaluation_150000 --output results/figures/comparison_seed0

Existing output directories are protected against overwrite. Use a new directory for each new run. To continue
the same run after an interruption:
  python -m experiments.run --resume results/data/baseline_seed0/checkpoint.npz --output results/data/baseline_seed0 --evaluate

The default is 150000 steps per condition, N=64, sigma=0.4, and l=0.5 versus 0.1.
Add --steps 2000 with a new output directory for a short installation check only (not a reproduction).
Additional paired seeds: add --seed 1 and use new directories for both conditions.

See README.md and references/implementation_notes.md for model details, paper/code discrepancies, analysis
conventions, and limits of interpretation. IMPORTANT: this implementation used the authors' public code (commit
940659e) to audit conventions and to choose the initialisation and optimiser settings; the NumPy implementation did
not. The implementation uses a specified norm tolerance (0.02) that differs from the public code. Check held-out
positivity and norm losses before declaring success. One training seed per condition is not a robustness study.

COMPLETED SEED-0 ATTEMPT
See results/reference_runs/README.md and results/figures/comparison_seed0. Both 150000-step runs completed, but did
not cleanly recover the intended hexagonal baseline; the baseline also missed its held-out norm target. Checkpoint
snapshots and evaluation figures are included under results/reference_runs.

BASELINE DIAGNOSTICS (3 conditions x 3 matched seeds, 150000 steps each; all nine complete)
See results/diagnostic_study/README.md. Neither isolated change resolved the baseline; no run met both common targets.
  python -m experiments.diagnostics plan   --output results/data/my_diagnostics
  python -m experiments.diagnostics run    --output results/data/my_diagnostics --workers 2
  python -m experiments.diagnostics report --output results/data/my_diagnostics
The report uses shared evaluation settings and common constraint tolerances. Read experiments/diagnostics.md before
interpreting the results. Retain all seeds and failed runs.

CROSS-CHECK WITH THE NUMPY IMPLEMENTATION  (needs ../numpy_impl)
experiments/fixtures/equation_contract holds shared inputs and independent NumPy expectations. To re-run the audit of
the NumPy core on those inputs (about 10 s; the output folder must not exist or be empty):
  python -m comparison.partner_equations --output results/data/partner_equation_recheck
Stored result: results/partner_comparison (loss terms agree to 2.8e-13, directional derivatives to 5.1e-11).

REVIEW
The merged review (both implementations) is ../review/main.tex and ../review/main.pdf.
