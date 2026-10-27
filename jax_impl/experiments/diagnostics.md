# Diagnose the baseline before extending the model

Question: do two documented numerical ambiguities explain the failure of the
initial single-module baseline? This is a sensitivity study of a specified
implementation, not a claim that any one convention is the authors' exact setup.

## Conditions fixed before inspecting the new results

All conditions use N=64, 31 frequencies, l=0.5, L=1 and the original 150,000-step
budget. Seeds 0, 1 and 2 are independent repetitions. Within each seed, the entire
initial state is identical and the same random sampling stream is used.

| Condition | Change from baseline | Reason |
| --- | --- | --- |
| baseline | None | Repeat the current sigma=0.4, norm_target=0.02 implementation |
| sigma_variance | sigma=sqrt(0.4) | Test treating 0.4 as the variance, as the public sigma_sq argument does |
| norm_table | norm_target=exp(4) | Test the literal table B.3 log target k_norm=4 under equation 26 |

The norm-table condition intentionally permits much larger norm deviations. It
does not reproduce the upstream raw-rate norm formula, aggregation or positivity
update. A pass against its own loose training target is not a biological success.
Every condition is ALSO assessed against the same original 0.02 norm tolerance
and exp(-9) negativity tolerance. These are numerical tolerances, not sufficient
criteria for a valid grid module.

Different sigma values define different functional objectives. Do not compare
their objective values as if a smaller number meant a better representation.
Compare held-out constraint errors, map morphology and descriptive grid scores.

## Run locally

From the repository root, choose an unused output directory:

```powershell
.\.venv\Scripts\python.exe -m experiments.diagnostics plan --output results/data/my_diagnostics
.\.venv\Scripts\python.exe -m experiments.diagnostics run --output results/data/my_diagnostics --workers 2
.\.venv\Scripts\python.exe -m experiments.diagnostics report --output results/data/my_diagnostics
```

The default plan contains nine full runs. Use `--workers 1` for less concurrent
CPU/memory load. Training logs are in `logs/`; checkpoints and metadata are in
`runs/`. Checkpoints at 25,000 and 75,000 steps are retained to inspect the
trajectory without selecting a favourable checkpoint as the final result.
Training histories also record the largest frequency component and multiplier
clipping. Final-checkpoint evaluation is independent of those intermediate maps.

Repeat the `run` command after an interruption: completed runs are retained;
partial runs resume with their optimiser and random state. Source files,
environment and configurations must still match. Do not edit implementation
files midway through a study. The plan checker rejects missing seeds and
undeclared additional parameter changes.

For a quick workflow check use `plan --steps 30 --seeds 0` in a separate
directory. Such runs verify plumbing only and must not be used to claim model
convergence. Existing output directories are protected against overwriting.

The report checks actual final checkpoints and matching training provenance.
It chooses one common map resolution sufficient for ALL learned frequencies,
with extent fixed at [-1.5,1.5] on both axes. If this exceeds 768 samples per axis,
it refuses to silently undersample; increase `--max-resolution` explicitly after
considering memory use. All populations are displayed in neuron-index order.

`runs.csv` gives each independent run, while `paired_differences.csv` subtracts
the baseline from each other condition within the same training seed. The plot
connects these paired measurements. Three seeds are a small robustness check;
no p-values, shuffled grid significance, independent-neuron error bars, or module
counts are claimed. Reports are written once, and a completion status is written
only after all evaluations succeed. A partial report is not a completed result.

## Comparing Tafara's and Nathan's implementations

The portable fixture in `experiments/fixtures/equation_contract/` contains exact
arrays, the configuration, a documented basis order, and six expected values.
Expected losses are calculated with explicit NumPy pairwise differences;
directional derivatives use central finite differences. The JAX implementation
is tested against them. Each implementation should read the saved arrays rather
than regenerate inputs from a seed (different RNG libraries can differ).

Produce a JSON file with the keys from `expected.json`:
`functional`, `positivity`, `norm`, `total`,
`weight_directional_derivative`, `frequency_directional_derivative`.
Derivatives are of the weighted total objective along the supplied unit
directions. Use float64 and preserve the original-room normaliser in shifted
rooms. Then compare:

```powershell
.\.venv\Scripts\python.exe -m experiments.equation_check --output experiments/fixtures/equation_contract --candidate path/to/other_implementation.json
```

The checker reports each mismatch and exits unsuccessfully on disagreement.
Agreement on this small contract supports equation-level agreement; it does not
prove the optimisers, sampling streams, or full results are equivalent.
Nathan's pinned implementation has now been checked on this fixture after it
became available on Git. See `results/partner_comparison/README.md` for the
measured agreement and the explicit functional-weight alignment. The training
and output-map comparisons remain distinct, unfinished questions.

## Decision after the study

The completed nine-run study is archived in `results/diagnostic_study`. Neither
isolated change established a credible baseline; no final run met both common
held-out targets. The shared-input audit passes after explicit weight alignment,
while the constructed-code control shows that finite sampling affects the norm
tolerance. Keep this negative result and calibrate conventions before extending
the spatial-scale experiment. The following decision rules remain applicable
to future studies.

1. Inspect all-cell maps and both constraint errors for every seed; do not select
   a condition solely because its score looks attractive.
2. If a justified convention gives a credible baseline across seeds, freeze that
   convention and repeat the original l=0.5 versus l=0.1 comparison with matched
   seeds. This is a NEW experiment, not a replacement for the original failed runs.
3. If none does, retain the negative finding and use the shared equation fixture
   and saved trajectories to distinguish implementation and optimisation issues.
4. Package a verified, documented example for NeuralPlayground before pursuing
   a full model integration. This repository currently makes no claim of an
   accepted NeuralPlayground contribution.
