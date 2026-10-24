# Comparing the two independent implementations

Compare the specified mathematical problems before comparing figures. A JAX
and NumPy implementation can be equivalent, but matching parameter names alone
does not establish that they optimise the same objective.

A ready-to-share numerical fixture is now in `fixtures/equation_contract/`.
See `diagnostics.md` for the input format and comparison command. It includes
independently calculated NumPy losses and finite-difference derivatives. This
enables Tafara and Nathan to compare equations without relying on RNG equivalence.

| Item | This branch's primary experiment |
| --- | --- |
| Population and frequencies | 64 neurons; 31 sine/cosine pairs plus the constant |
| Occupancy | Gaussian, L=1; 150 sampled positions |
| Comparison | l=0.5 versus l=0.1, everything else fixed |
| Neural scale | sigma=0.4, so the kernel variance is 0.16 |
| Rate normalisation | Per-neuron L2 SUM over the original room, not RMS |
| Shifted rooms | 15 Gaussian shifts, standard deviation 3; original-room normaliser reused |
| Positivity penalty | Mean negative part of normalised shifted-room rates |
| Norm penalty | Mean squared error of each shifted-room sum of squared rates from 1 |
| Constraint targets | Positivity exp(-9); norm 0.02 (explicit implementation choice) |
| Functional scaling | 25 times the M-by-M pairwise mean |
| Optimiser | Adam, learning rates 0.1, both betas 0.9, epsilon inside square root |
| Constraint updates | GECO, smoothing 0.9, rate 1e-4; next-step multipliers |
| Sampling and budget | Resample every 5 steps; 150,000 steps per condition |
| Randomness | One JAX seed controls initialisation and sampling; matched within the pair |
| Result selection | Final checkpoint, every neuron in index order |

The rationale and deviations from the paper/public code are in
`references/implementation_notes.md`. In particular, check whether the other
implementation uses `sigma` or `sigma_sq`, and whether its norm penalty is
actually evaluated on normalised rates. Those differences can change the result
even when both implementations have correct gradients.

For a numerical comparison, evaluate both implementations on an identical small
set of positions, shifts, weights, and frequencies. Compare each unweighted loss
term and a directional derivative before attempting full-run agreement. Different
random number generators need not produce identical seeds or trajectories.

For a result comparison, use the same physical map extent, resolution,
autocorrelation convention, annuli, and held-out sampling distributions. Report
negative activity and norm errors next to the maps. Positive grid scores alone
do not establish correct constraint enforcement or a complete grid-cell module.

Share the run's `config.json`, `metadata.json`, `history.jsonl`, final
`checkpoint.npz`, and evaluation folder. These are local under `results/data`
and excluded from Git; compact figures and summaries are versioned separately.
Keep initial and failed runs. Treat additional training seeds as repetitions;
neurons within one trained population are not independent repetitions.
