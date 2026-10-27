# Completed baseline diagnostics

All nine planned runs completed 150,000 updates: three conditions at each of
seeds 0, 1 and 2. The plan was fixed before inspecting these results. Within a
seed, initial states match exactly; training source hashes and package versions
match across runs. See `report/README.md` and `report/runs.csv` for every result.

## Findings

| Setting changed from baseline | Median grid-score range | Held-out norm-error range | Both common targets met |
| --- | --- | --- | --- |
| None | 0.0483 to 0.0645 | 0.02086 to 0.03011 | 0/3 |
| sigma=sqrt(0.4) instead of 0.4 | 0.0666 to 0.1006 | 0.01483 to 0.02435 | 0/3 |
| norm_target=exp(4) instead of 0.02 | -0.1226 to -0.0366 | 1.7653 to 3.8727 | 0/3 |

The sigma change increased the descriptive score in each matched pair, but
does not establish a convincing hexagonal baseline. The relaxed norm target
substantially changed the learned frequencies and allowed much larger room
energy errors. Two relaxed-target runs met their own training tolerances, which
is not equivalent to meeting the common assessment or producing valid grids.
None of the nine passed both the original 0.02 norm and exp(-9) mean-negativity
targets on held-out samples. These targets are implementation choices, not a
definition of biological grid cells; the constructed control in
`../constraint_control` demonstrates why the norm threshold needs calibration.

The nine conditions share a 352-by-352 evaluation grid on [-1.5,1.5]^2, with the
same held-out samples. The original spatial-scale comparison used 320-by-320;
its descriptive grid scores should not be mixed with these new values.
The new baseline seed-0 checkpoint exactly matches the original training state,
confirming that added recording did not change that trajectory.

This is a three-seed sensitivity study, not a test of every plausible numerical
convention. The changed sigma also changes the objective; objective magnitudes
are therefore not compared across conditions. Neither an irreproducibility
claim nor a successful reproduction follows from these results. The justified
next step is calibration and reconciliation of numerical conventions before
extending the spatial-scale experiment, rather than selecting a favourable seed.

## Evidence and local reuse

- `plan.json` and `configs/`: frozen settings.
- `runs/`: initial/final states, 25k/75k snapshots, logs and metadata.
- `report/`: common evaluation settings, all-neuron figures, summaries and paired CSV.
- `archive_manifest.json`: SHA256 checksums of all 133 archived evidence files.

Large regenerable `maps.npz` arrays are omitted. Re-evaluate any saved checkpoint
with `python -m experiments.evaluate PATH --output NEW_DIRECTORY --resolution 352
--extent 1.5`, using the remaining settings in `report/evaluation_settings.json`.
For a new complete study follow `../../experiments/diagnostics.md`; use a fresh
output directory. Archive completed results with
`python -m comparison.archive_study SOURCE --output NEW_ARCHIVE`.

The nine recorded training durations sum to about 71.2 minutes of wall time
across processes, with two runs allowed concurrently. This is not elapsed study
time or CPU-hours. Evaluation time and preliminary experiments are additional.
Hardware and software details are preserved in each run's metadata.
