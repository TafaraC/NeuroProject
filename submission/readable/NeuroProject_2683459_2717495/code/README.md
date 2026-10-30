# Code and saved inputs

| Folder or file | Contents |
|---|---|
| `numpy_impl/src/actionable/` | NumPy model, gradients, analysis and module-angle calculation |
| `jax_impl/src/` | JAX Fourier model, optimiser and evaluator |
| Each implementation's `experiments/` | Configurations and experiment entry points |
| Each implementation's `tests/` | Numerical, geometry and continuation tests |
| Each implementation's `results/` | Original parameters, settings and records read by the analysis scripts |
| `jax_impl/comparison/` | Shared-equation comparison, checkpoint conversion and constructed-code control |
| `review/` | Original figure and numerical-table generation scripts |
| `smoke_test.py` | Combined installation check using temporary outputs |

Start with the package's `README.txt`, then follow [RUN_GUIDE.md](RUN_GUIDE.md). Use the top-level `run.py` command interface rather than guessing which internal module is executable. Its commands have `--help` and leave the saved evidence untouched. Internal libraries and fixed-purpose analysis scripts are preserved byte-for-byte, including their original command behaviour; they are not all standalone command-line tools.

The detailed result folders are retained because the unchanged code reads their paths. In particular, number-generation code reads NumPy training logs, the checkpoint comparison reads `SNAPSHOT.json`, and the equation test reads fixture manifests. Those files are functional inputs, not the primary explanation of the results. Read the tables in `../evidence/` for that explanation.

## Numerical conventions

| Choice | NumPy | JAX reference runs |
|---|---|---|
| Functional-loss weight | 1 | 25 |
| Norm target | `exp(4)` in main runs; `exp(-4)` sensitivity runs | 0.02 baseline; `exp(4)` diagnostic condition |
| Frequency initialisation | Gaussian, standard deviation 4 | Uniform on [0, 2) |
| Coefficient initialisation | Normal, scaled by feature count | Normal, standard deviation 1 |
| Adam denominator | `sqrt(v) + eps` | `sqrt(v + eps)` |
| Main grid assessment | Period-scaled analytic autocorrelation | Fixed-window Pearson autocorrelation |

Both use sampled sum normalisation. In the shared fixture, the functional weight is explicitly aligned before comparing derivatives. Agreement checks equations, not training equivalence or successful convergence. The two grid-score methods should not be treated as interchangeable measures.
