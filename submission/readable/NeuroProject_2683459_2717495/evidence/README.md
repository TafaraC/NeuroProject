# Evidence guide

Start with the report tables, then use the following CSVs to inspect the underlying values. CSV files open in a spreadsheet; numeric values have not been rounded during export.

| File | One row represents | Use |
|---|---|---|
| `numpy_runs.csv` | A saved NumPy run folder, its configuration and scalar results | Find any main run, ablation, sweep or sensitivity result |
| `numpy_modules.csv` | One detected module within a NumPy run | Inspect module geometry and size |
| `jax_diagnostics.csv` | One of nine JAX diagnostic runs | Compare three matched seeds across three conditions |
| `common_evaluation.csv` | One of five NumPy checkpoints measured by the JAX evaluator | Reproduce report Table A3 |
| `sampling_summary.csv` | One sampling budget in the constructed-code control | Compare 50, 150 and 600 sampled positions |
| `sampling_batches.csv` | One sampling batch from that control | Inspect uncertainty across 128 batches per budget |
| `training_history.csv` | One recorded optimisation point or block | Inspect changes during training, not just endpoints |
| `numpy_aggregate_results.csv` | One scalar entry in the original aggregate tables | Trace scaling, angle, ablation and sensitivity statistics |

## Find a reported result

1. **Choose the table.** For the main NumPy runs, open `numpy_runs.csv` and filter `original_run` for `main_sigma`.
2. **Choose the setting.** Filter `setting_sigma`, `setting_seed` and `setting_steps`. Read `result_n_hex_modules`, `result_n_modules` and `result_frac_grid_analytic`.
3. **Follow the source.** Prefix `original_run` with `code/` to find the saved configuration, metrics and parameters. For aggregate tables, `measurement` gives a slash-separated dictionary/list location in the original JSON.
4. **Repeat the calculation if needed.** Use `python run.py compare`, `python run.py control` or the experiment commands in `../code/RUN_GUIDE.md`. New outputs are separate from the supplied evidence.

## Conditions and protocol

| Saved condition | Meaning in the report |
|---|---|
| `baseline` | JAX sigma 0.4, spatial scale 0.5, norm target 0.02 |
| `sigma_variance` | Same configuration except sigma is the square root of 0.4 |
| `norm_table` | Same configuration except norm target is exp(4) |
| `main_sigma1.0_seed0/1` | NumPy large-sigma runs, 150,000 updates |
| `main_sigma0.2_seed0/1/2` | NumPy smaller-sigma runs, 150,000 updates |
| `sweep_sigma0.4_seed0/1` | NumPy sigma 0.4 sweep runs, 50,000 updates |
| `sens_kn-4_sigma1.0` | NumPy tighter norm target exp(-4), 50,000 updates |

The nine JAX diagnostic runs use seeds 0, 1 and 2, each trained for 150,000 updates. Initial states match within seed. Common evaluation uses eight held-out batches, seed 12345, and a 352-by-352 grid over [-1.5, 1.5] on both axes. NumPy training budgets vary by experiment and are recorded in `setting_steps`.

## Column guide

- `original_run` identifies the original folder or data file, relative to `code/`.
- `setting_*` columns are training configurations; `result_*` columns are saved measurements. `setting_N`, `setting_D`, `setting_M` and `setting_MS` mean neurons, frequency pairs, sampled positions and shifts.
- `sigma` is neural separation scale; `l` is spatial discrimination scale. They are distinct quantities.
- `frac_grid_analytic` is the fraction of cells above 0.3 under NumPy's analytic grid-score measure. It is not a probability of statistically significant grid structure.
- `grid_score_median` in the common and JAX tables uses a fixed spatial window. It is not directly comparable with the analytic score.
- `norm_loss` is mean squared shifted-room norm error. `positivity_loss` is the mean negative-activity penalty. `negative_fraction_below_minus_1e_minus6` counts rates below that numerical threshold.
- `training_targets_met` uses each run's training thresholds; `common_targets_met` uses the baseline thresholds, including for the relaxed-target condition.
- `record_index` orders history entries. A blank cell means that implementation did not record the field; it does not mean zero. JAX `step` identifies an update count and `block_seconds` records elapsed block duration. NumPy history names are preserved from its saved arrays.
- In `numpy_aggregate_results.csv`, `measurement` is the original nested key. Numeric `value` cells preserve the original value; text values use quoted JSON strings and `value_type` distinguishes them.

## Reconstruct the headline comparisons

- For Table A1's large-sigma result, filter NumPy runs to sigma 1.0, seeds 0 and 1 and 150,000 steps: each has one hexagonal module. At sigma 0.2, the three main seeds have none.
- For Table A2's JAX norm ranges, group `jax_diagnostics.csv` by `condition` and take minimum and maximum `norm_loss`. Round only when displaying the report values.
- Table A3 is `common_evaluation.csv` in its existing row order, rounded to four decimals for grid score and norm loss and seven for positivity loss.
- Figure A6's means and standard errors come directly from `sampling_summary.csv`. The batches are repeated samples of one fixed constructed code, not independent training runs.

## Limits

There are 60 completed NumPy folders but 59 distinct final parameter arrays; `sweepl02_sigma0.4_seed0` duplicates `sweepl_l0.2_sigma0.4`. Do not count them as independent replicates. Settings use one to nine seeds, with unequal budgets. Neurons and modules from the same training run are not independent training seeds. The reported scaling interval is a module-level bootstrap from the saved analysis, not an independent-run confidence interval.

Many original records remain under `code/` because scripts read them by their original names. Machine paths recorded inside run metadata are provenance fields, not paths a reader must create. Use this guide instead of those metadata paths. The original training code and the arrays are unchanged.
