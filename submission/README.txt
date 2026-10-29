COMPUTATIONAL NEUROSCIENCE: FIRST-DEADLINE SUBMISSION
Tafara Chibanda (2683459) and Nathan Le Roux (2717495)
Deadline: 27 October 2026, 17:00, South Africa time.

START HERE
Review.pdf is the joint paper review, followed by supporting evidence and the
NeurIPS checklist. Contribution_statement.pdf describes the independent work.
Tafara will append the completed official Faculty AI Declaration in LaTeX before
uploading. This is the remaining author-supplied item; do not omit it.
No NeuralPlayground integration is claimed or required for this first deadline.

CONTENTS
src/, experiments/, comparison/, tests/: Tafara's JAX implementation and checks.
nathan/: Nathan's independent NumPy implementation and all saved experiments.
results/: Tafara's saved evidence and common evaluations of Nathan's checkpoints.
review/paper_review/: editable review source, template and figures.
requirements.txt: pinned scientific runtime, sufficient for both implementations.
requirements-test.txt: runtime plus pytest for Tafara's test suite.
MANIFEST.json: file hashes for checking the archive after extraction.
PROVENANCE.json: source commits and packaging information.

INSTALL (Windows PowerShell, from this extracted directory)
Use 64-bit Python 3.14; verified locally with Python 3.14.3 on Windows x64.
Internet access is required for the initial dependency installation.
  py -3.14 -m venv .venv
  .\.venv\Scripts\python.exe -m pip install -r requirements-test.txt
  .\.venv\Scripts\python.exe -m pip check
On another platform, use its Python 3.14 executable and .venv/bin/python instead.
Other operating systems have not been tested. GPU support is not required.

VERIFY THE CODE (no long experiment needed)
  .\.venv\Scripts\python.exe -m pytest tests -q
  .\.venv\Scripts\python.exe -m unittest discover -s nathan/tests -v

SHARED EQUATION CHECK (no Git or network needed)
  .\.venv\Scripts\python.exe -m comparison.partner_equations --source nathan/src/actionable/core.py --output results/data/equation_check
The adapter enforces the reviewed source hash. The old pinned commit in its
output identifies the first audited core; Nathan's packaged newer commit has
the same core bytes. The functional coefficient is explicitly aligned (25 vs 1).

TAFARA: SHORT EXECUTION CHECK
  .\.venv\Scripts\python.exe -m experiments.run --config experiments/configs/baseline.json --output results/data/smoke --steps 30
This is a plumbing check, not a reproduction result. Use unused output paths.

TAFARA: FULL RUNS
  .\.venv\Scripts\python.exe -m experiments.run --config experiments/configs/baseline.json --output results/data/new_baseline --evaluate
  .\.venv\Scripts\python.exe -m experiments.run --config experiments/configs/fine_resolution.json --output results/data/new_fine --evaluate
The default budget is 150000 steps each. The original concurrent runs took about
9.3 minutes each on the recorded machine; other machines will differ. If a new
run's frequencies require a larger plotting grid, the evaluator reports the
minimum resolution. Keep shared evaluation settings when comparing conditions.

TAFARA: THREE-SEED DIAGNOSTIC STUDY
  .\.venv\Scripts\python.exe -m experiments.diagnostics plan --output results/data/new_diagnostics
  .\.venv\Scripts\python.exe -m experiments.diagnostics run --output results/data/new_diagnostics --workers 2
  .\.venv\Scripts\python.exe -m experiments.diagnostics report --output results/data/new_diagnostics
This runs nine 150000-step experiments. Retain all seeds, including failures.

NATHAN: RUN INDEPENDENTLY
All 60 saved experiment folders are included; 59 have distinct parameter sets.
There is one duplicated spatial-scale run, documented in the comparison report.
From nathan/, run with the same environment:
  ..\.venv\Scripts\python.exe experiments/run_all.py --quick --only main
This writes separate results_quick/ folders and performs short checks.
For full experiments, experiments/run_all.py --only main uses existing saved
results and skips completed runs. To train afresh without overwriting evidence,
use the direct experiment command from nathan/ with a separate results path:
  $env:ACTIONABLE_RESULTS = 'results_new'
  ..\.venv\Scripts\python.exe experiments/exp1_grid_emergence.py --job main_sigma1.0_seed0
  Remove-Item Env:ACTIONABLE_RESULTS
See nathan/experiments/jobs.py for all named jobs. Nathan's requirements.txt is
the original unpinned file; use the pinned root requirements for this package.

COMPARE SAVED RESULTS WITHOUT RETRAINING
  .\.venv\Scripts\python.exe -m comparison.partner_results --snapshot nathan --output results/data/common_recheck
This re-evaluates the five declared Nathan checkpoints with the same grid and
held-out samples used for Tafara's diagnostic study. It takes several minutes.
It does not claim the optimisers or initial arrays were matched.

RESULTS TO READ
  results/reference_runs/README.md: original two spatial-scale attempts.
  results/diagnostic_study/README.md: all nine controlled diagnostic runs.
  results/constraint_control/README.md: constructed-code sampling calibration.
  results/partner_comparison/README.md: independent equation audit.
  results/partner_results_20261026/README.md: joint numerical comparison.
  nathan/results/tables/: Nathan's native analysis summaries.
The review distinguishes positive grid geometry, constraint residuals and
analysis-window effects. Neurons are not independent training repetitions.

REBUILD THE REPORT
Edit review/paper_review/main.tex. Add the official AI Declaration through
review/paper_review/ai_declaration.tex. Compile main.tex with Tectonic or another
compatible LaTeX distribution, then replace the root Review.pdf with main.pdf.
The supplied ccn.sty is preserved. The main review fits within three pages;
supporting figures, tables, references and checklist are additional material.
If editing an extracted archive, regenerate MANIFEST.json after final changes
using: python submission/verify_manifest.py --refresh
Then zip the folder. The automated builder is for the development Git checkout.

FIRST-DEADLINE SCOPE
The paper review, full implementations, requirements, execution instructions,
saved numerical evidence, contribution statement and checklist are included.
Both authors confirmed the judgements and contributions on 26 October 2026.
The authors are responsible for appending their completed official declaration
and uploading the final files to the course submission portal.
