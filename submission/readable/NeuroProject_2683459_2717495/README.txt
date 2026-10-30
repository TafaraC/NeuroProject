GRID CELLS FROM MINIMAL CONSTRAINTS
Tafara Chibanda (2683459) and Nathan Le Roux (2717495)

Read NeuroProject_Report.pdf first. It reviews Dorrell et al. (2023),
Actionable Neural Representations: Grid Cells from Minimal Constraints.
Its main review occupies three pages; evidence and the NeurIPS checklist follow.

CONTENTS
NeuroProject_Report.pdf       Review, supporting tables and figures, checklist
Contribution_statement.pdf   Contribution statement
CONTRIBUTIONS.md              Roles, tools, reused resources and data provenance
VALIDATION.md                 Checks performed on this package and their results
code/                        Both implementations, tests and saved inputs
evidence/                    Tables for inspecting results without running code
declarations/                Signed Faculty AI declarations of both authors
run.py                       Common command interface; run python run.py --help
requirements.txt             Runtime and test packages
requirements-pinned.txt      Recorded scientific runtime versions

RESULTS
- NumPy found a single hexagonal module in both full sigma=1.0 runs.
- Neither implementation consistently recovered the selected multi-module
  or sigma=0.4 hexagonal examples under its documented training conventions.
- The implementations agree on a shared loss/derivative fixture. Their training
  settings differ, so their outcomes are not a controlled ranking.
- Five checkpoints re-evaluated on a fixed window show why the observation
  window matters for broad grids. The sampling control shows why norm targets
  require calibration.

INSTALL AND CHECK
1. Open a terminal in this folder. The setup has been tested with Python 3.14
   on Windows and Python 3.12 on Linux. Installation needs internet access,
   but no account or dataset.

2. Create and activate an environment.
   Windows PowerShell:
     py -3.14 -m venv .venv
     .\.venv\Scripts\Activate.ps1
   Linux/macOS:
     python3 -m venv .venv          (Python 3.12 or later)
     source .venv/bin/activate
   If PowerShell blocks activation, run .\.venv\Scripts\python.exe in place
   of python below. macOS has not been tested.

3. Install the recorded scientific packages and test runner:
     python -m pip install -r requirements-pinned.txt
     python -m pip install -r requirements.txt
     python -m pip check
   requirements.txt alone permits compatible newer packages. To repeat the
   recorded test setup, use pytest==9.1.1 with the scientific pins.
   The model runs on CPU; no GPU or PyTorch installation is needed.

4. Check the installation:
     python run.py tests
   Expected: 20 NumPy tests and 15 JAX tests pass. For tests plus short training
   and a loss/derivative comparison:
     python run.py smoke
   Short jobs validate the workflow, not the scientific conclusions.

5. Find or reproduce a result using evidence/README.md and code/RUN_GUIDE.md.
   Saved inputs stay under code/. New runs go to reproduced_results/.
   Each command explains its options with --help. The first JAX run includes
   compilation time. Some commands require a fresh output directory: rename
   a previous reproduced_results subfolder before repeating it.

REPORT AND DECLARATIONS
The PDF is ready to read without LaTeX. Editable report sources and the package
builder are supplied separately next to this package in the repository's
submission/readable/ folder. They are not needed to run the experiments.
The signed Faculty AI declarations of both authors are in declarations/.

VERIFICATION AND LIMITS
See VALIDATION.md for the checks actually performed on this reorganised copy.
Synthetic positions are generated from recorded seeds; no private data is used.
Existing model code, parameters, configurations and recorded numeric results
are preserved. Evidence CSVs organise those values without rounding them.
