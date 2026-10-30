# JAX implementation

This implementation uses automatic differentiation. The reference configuration uses 64 neurons, 31 frequency pairs, sigma 0.4, 150 sampled positions and 15 shifts. It applies functional-loss weight 25 and norm target 0.02. These are documented numerical choices, not a claim to reproduce every convention of the authors' code.

| Folder | Contents |
|---|---|
| `src/` | Fourier responses, optimiser, checkpointing and evaluation |
| `experiments/` | Reference configurations, shared fixture and diagnostic workflow |
| `comparison/` | Equation comparison, common evaluation and constructed-code control |
| `tests/` | Fifteen tests of model identities, evaluation and diagnostics |
| `results/reference_runs/` | Two spatial-scale reference runs |
| `results/diagnostic_study/` | Nine matched-seed diagnostic runs |
| `results/constraint_control/` | Fixed-code sampling experiment |
| `results/partner_results_20261026/` | Common evaluation of five NumPy checkpoints; legacy directory key retained for code compatibility |

Follow `../RUN_GUIDE.md` from the package root. Read `../../evidence/README.md` for condition names and result interpretation. Stored metadata can contain the machine paths used during training; those paths do not need to exist on another computer.
