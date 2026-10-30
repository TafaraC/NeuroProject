# NumPy implementation

This implementation uses hand-derived gradients, checked against finite differences. Its main runs use a functional-loss weight of 1 and a norm target of exp(4). Sensitivity experiments vary those choices.

| Folder | Contents |
|---|---|
| `src/actionable/` | Model, gradients, optimiser, geometry and module-angle analysis |
| `experiments/` | Training jobs, sweeps, ablations, plots and numerical tables |
| `tests/` | Twenty tests of gradients, geometry and exact continuation |
| `results/runs/` | Saved parameters, configurations, metrics and training records |
| `results/tables/` | Aggregate experiment measurements |
| `results/figures/` | Saved plots named by the snapshot manifest |
| `review/tables/` | Numerical macro inputs read by the combined report-generation script |

Follow `../RUN_GUIDE.md` from the package root. The 60 saved run folders include one duplicate final parameter array; see `../../evidence/README.md`. `SNAPSHOT.json` is required by the checkpoint-comparison command and checks the saved bytes against the supplied snapshot.
