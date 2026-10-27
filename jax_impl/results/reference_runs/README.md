# Completed independent reproduction attempt -- 24 October 2026

Both paired seed-0 runs completed 150,000 steps on CPU. Training used commit
`dca93cb`, with full source hashes and resolved settings in each run's metadata.
The later analysis correction changes map sampling only, not training or the
held-out constraint calculations. The evaluation summary records its source hash.

**The final checkpoints do not cleanly reproduce the intended hexagonal baseline
or establish the paper's predicted change with l.** They contain strongly
elongated periodic patterns, and both median sixfold grid scores are near zero.
This is a documented reproduction attempt, not evidence against the paper's
optimality claim: the optimiser, initialisation, and specified norm tolerance
can affect which solution is found.

| Final held-out measurement | l = 0.5 | l = 0.1 |
| --- | ---: | ---: |
| Median descriptive grid score | 0.0503 | 0.00884 |
| Mean negative firing penalty | 0.0000996 | 0.0001106 |
| Mean squared shifted-room norm error | 0.03011 | 0.01950 |
| Mean negativity target met? | Yes | Yes |
| Norm target 0.02 met? | No | Yes |
| Fraction of sampled rates below -0.000001 | 3.61% | 5.08% |
| 95th percentile absolute room-energy error | 0.339 | 0.279 |
| Relative numerical translation error | 0.0000233 | 0.000197 |

Passing a mean negativity tolerance is not exact non-negativity. The nonzero
negative fractions and room-energy deviations must be reported. The weight
matrices remain full column rank under the reported numerical rank convention,
but are poorly conditioned (approximately 97,000 and 316,000), which limits
numerical translation accuracy. Cells within each population are not independent
replicates. Only one training seed was run per condition.

## Analysis correction

The initial 96-by-96 view over [-5,5] was too coarse for the learned frequencies
(maximum axis frequencies approximately 136 and 100 radians per spatial unit).
It was rejected during visual/spectral inspection. Final maps and scores use a
shared 320-by-320 view over [-1.5,1.5], with at least four samples per shortest
axis wavelength. The evaluator now enforces this check, covered by a regression
test. Earlier coarse-grid outputs remain marked SUPERSEDED in the ignored local
run folders and are not included in this bundle. No training run was changed or
selected based on this correction.

The final figures are in `results/figures/comparison_seed0`. Each subdirectory
here includes initial and final checkpoints, resolved configuration, metadata,
block-end training history, status, and final evaluation summaries/figures.
Large numerical map arrays are reproducible and remain in the ignored local
`results/data` folders.

## Re-evaluate without retraining

From the repository root, using a NEW output directory:

```powershell
.\.venv\Scripts\python.exe -m experiments.evaluate results/reference_runs/baseline_seed0/checkpoint.npz --output results/data/baseline_recheck
.\.venv\Scripts\python.exe -m experiments.evaluate results/reference_runs/fine_seed0/checkpoint.npz --output results/data/fine_recheck
```

Full training commands are in the main README. The saved metadata is preserved
from training, so resume correctly rejects subsequent source changes; use the
original implementation revision for exact continuation, or start a fresh run
with the current code. These runs already reached their configured budget.

Nine automated tests pass: independent loss evaluation, directional finite
differences, actionability and composition, GECO response, exact checkpoint
continuation, configuration validation, synthetic pattern scoring, constant-map
handling, and rejection of undersampled map grids. A separate command-line
pause/resume check passed as well. Test success verifies implementation details;
it does not establish that optimisation reached the predicted solution.

For the later comparison with the other implementation, start with the
normalisation and sigma/norm-loss conventions in
`experiments/comparing_implementations.md` and
`references/implementation_notes.md`. In particular, norm_target=0.02 is our
explicit numerical choice, not a claimed parameter from the published Figure 8.
