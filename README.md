# Dorrell et al. (2023) Reproduction

Computational Neuroscience project by Nathan Le Roux and Tafara Chibanda.

Paper: **Actionable Neural Representations: Grid Cells from Minimal Constraints**
(ICLR 2023), https://arxiv.org/abs/2209.15563.

## Scope and current status

Independent JAX implementation on `codex/independent-reproduction`: a learned
Fourier representation, the paper's functional and biological losses, Adam/GECO
optimisation, resumable experiments, and held-out analysis. The target is a
single-module grid-cell reproduction and a comparison of l=0.5 with l=0.1 at
fixed N=64 and sigma=0.4. Repeated seeds are optional.

Read `references/implementation_notes.md` before interpreting results: the
paper and public implementation differ in their norm loss and sigma convention.
The chosen norm tolerance and other numerical choices are explicitly documented.
Training completion alone does not establish successful reproduction.

**First completed result:** both seed-0 conditions ran for 150,000 steps, but did
not cleanly recover the intended hexagonal baseline. The baseline missed the
held-out norm target. See `results/reference_runs/README.md` for the measurements,
limitations, analysis correction, and saved checkpoints, and
`results/figures/comparison_seed0` for the comparison figures and CSV.

## Environment

Use **64-bit Python 3.14** (local setup: 3.14.3) with a project-local `.venv`.
The baseline environment uses **JAX on CPU**, NumPy, SciPy, Matplotlib, and Optax.
JupyterLab and IPython kernel support are available with the development tools,
along with pytest and Ruff for later validation and code checking.

This is a modern implementation environment, not the authors' original software
snapshot. Their code may need API adaptations. Native Windows JAX does not offer
NVIDIA GPU support; a GPU environment would need separate setup and verification.
See https://docs.jax.dev/en/latest/installation.html.
A short local benchmark measured roughly three seconds per 1,000 training steps;
full runs and concurrent workloads can differ. Default runs use 150,000 steps.

Setup verified on 11 October 2026: dependency consistency, JAX CPU compilation
and automatic differentiation, an Optax update, SciPy correlation, Matplotlib
image output, and execution through the local Jupyter kernel all passed.

### Set up another Windows checkout

Run these commands from the repository root in PowerShell:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip check
```

For only the scientific runtime, install `requirements.txt` instead.
The requirements files pin both direct and transitive dependencies. The `.in`
files list the intended direct dependencies for deliberate future updates;
install from the pinned `.txt` files for a matching environment.
This setup is verified on Windows x64; other platforms require their own check.

### Work in the existing environment

Activation is optional. Calling the environment's Python directly avoids
PowerShell activation-policy issues:

```powershell
.\.venv\Scripts\python.exe -m jupyterlab
```

Select the **NeuroProject (.venv)** kernel in JupyterLab. In an editor, select
`.venv\Scripts\python.exe` as the interpreter (a VS Code default is provided).
On a fresh checkout, add the named kernel within the environment:

```powershell
.\.venv\Scripts\python.exe -m ipykernel install --sys-prefix --name neuroproject --display-name "NeuroProject (.venv)"
```

If desired, activate with `.\.venv\Scripts\Activate.ps1` and use `python`.
No global Python packages or global notebook kernels are required.

## Repository layout

| Location | Purpose |
| --- | --- |
| `src/` | Fourier model, configuration, training, and analysis |
| `experiments/` | Run, evaluate, compare commands and JSON configurations |
| `notebooks/` | Future exploratory notebooks |
| `tests/` | Equation, gradient, actionability, resume, and metric checks |
| `references/` | Source links and reading notes |
| `results/data/` | Self-contained run folders, including checkpoints/logs/maps, excluded from Git |
| `results/checkpoints/` | Saved parameters, excluded from Git |
| `results/logs/` | Run records, excluded from Git |
| `results/figures/` | Selected figures suitable for version control |
| `results/reference_runs/` | Versioned final/initial checkpoint snapshots, settings, logs, and evaluation summaries |
| `review/` | Contribution statement and review source |
| `review/figures/` | Figures included in the report |
| `tmp/` | Disposable local setup and inspection files, excluded from Git |

The supplied paper, assignment PDF, and review-template ZIP remain at the root.
The contribution statement is in `review/contribution_statement.tex`.

## Reproducibility and submission

Run metadata automatically records the seed, resolved settings, code revision,
source hashes, dependency versions, and hardware. Evaluation records held-out
constraint checks. Keep unsuccessful runs and explain deviations from the paper.

The final submission will also need the review PDF, required checklist, and AI
declaration. This implementation does not write or revise the review for you.
`README.txt` contains a compact reproduction guide for submission.

## Run the reproduction

Run from the repository root. Keep output names unique: commands refuse to
overwrite previous experiments. Both configurations use the same seed so only
the spatial discrimination setting differs.

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe -m experiments.run --config experiments/configs/baseline.json --output results/data/baseline_seed0 --evaluate
.\.venv\Scripts\python.exe -m experiments.run --config experiments/configs/fine_resolution.json --output results/data/fine_seed0 --evaluate
.\.venv\Scripts\python.exe -m experiments.compare results/data/baseline_seed0/evaluation_150000 results/data/fine_seed0/evaluation_150000 --output results/figures/comparison_seed0
```

For a quick check, use a fresh output name and add `--steps 2000 --evaluate`.
This changes the total budget and is NOT a complete reproduction. To pause a full
run without changing its budget, use `--stop-at 10000`. Checkpoints are written
after each 1,000-step block (including Adam, GECO, RNG, and current samples).

```powershell
.\.venv\Scripts\python.exe -m experiments.run --resume results/data/baseline_seed0/checkpoint.npz --output results/data/baseline_seed0 --evaluate
```

Resume refuses changes to the recorded source or package versions. Abruptly
interrupting within a block loses only that block's work. On resume, the history
is trimmed to the last saved checkpoint. For an additional seed, use `--seed 1`
with a new output directory for EACH condition; do not treat neurons within a
run as independent experimental replicates.

To analyse a checkpoint separately (including the retained initial state):

```powershell
.\.venv\Scripts\python.exe -m experiments.evaluate results/data/baseline_seed0/initial.npz --output results/data/baseline_seed0/evaluation_initial
```

Each evaluation contains `summary.json`, numerical maps, all-neuron rate maps,
spatial autocorrelations, and a frequency plot. The comparison exports a CSV,
two figures, and a short result note. Read the positivity and norm checks before
calling a run successful; positive grid scores alone are insufficient.

Evaluation defaults to a 320-by-320 view of [-1.5,1.5] in each spatial coordinate.
It rejects grids that undersample learned frequencies. If this happens, rerun
`experiments.evaluate` with the reported minimum `--resolution`, using the SAME
resolution and extent for both conditions. The completed training checkpoint is
retained even if automatic evaluation cannot use the default grid.

For comparing this branch with your groupmate's implementation, see
`experiments/comparing_implementations.md`.
