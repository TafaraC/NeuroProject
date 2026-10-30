# Run guide

Run these commands from the package root after installing the packages in `README.txt`.

| Purpose | Command | Output |
|---|---|---|
| All 35 tests | `python run.py tests` | Terminal summary |
| Tests and short workflow checks | `python run.py smoke` | Temporary outputs; terminal summary |
| Shared losses and derivatives | `python run.py equations` | `reproduced_results/equations/` |
| Fixed-window comparison of five saved checkpoints | `python run.py compare` | `reproduced_results/common_evaluation/` |
| Constructed-code sampling control | `python run.py control` | `reproduced_results/sampling_control/` |
| Full NumPy single-module run | `python run.py numpy --job main_sigma1.0_seed0` | `reproduced_results/numpy/` |
| Short NumPy workflow check | `python run.py numpy --quick` | `reproduced_results/numpy/` |
| Full JAX baseline | `python run.py jax --condition baseline` | `reproduced_results/jax_baseline/` |
| Full JAX small-spatial-scale run | `python run.py jax --condition fine_resolution` | `reproduced_results/jax_fine_resolution/` |
| Regenerate report numerical macros | `python run.py numbers` | `code/review/` and `code/numpy_impl/review/tables/` |

## Nine controlled JAX runs

1. **Create the plan:** `python run.py diagnostics plan` creates three conditions with seeds 0, 1 and 2, each with 150,000 updates.
2. **Run the plan:** `python run.py diagnostics run --workers 2` runs at most two training processes together.
3. **Evaluate the checkpoints:** `python run.py diagnostics report` writes the common-grid summaries and plots.

The output is `reproduced_results/diagnostics/`. Initial states are matched within seed. The `baseline`, `sigma_variance` and `norm_table` names are defined in the evidence README. The two diagnostic interventions change one numerical choice at a time.

## Other NumPy experiments

1. **Find a job:** open `code/numpy_impl/experiments/jobs.py`. Its `JOBS` and `GROUPS` mappings list the sweeps, ablations, sensitivity settings and seeds.
2. **Run that job:** use `python run.py numpy --job JOB_NAME`. For example, `main_sigma0.2_seed1` runs the lower-sigma condition at seed 1.
3. **Analyse a complete campaign:** the original `experiments/run_all.py --analyse-only` command regenerates analysis using its existing `results/` folder. To preserve the supplied records, make a separate copy of `code/numpy_impl/` before using that original command. `python experiments/run_all.py --help` lists campaign options.

## Interpretation and troubleshooting

- Full reference runs use 150,000 updates; several NumPy sweeps and sensitivities use 50,000 or 30,000. A short run does not reproduce those results.
- Choose a fresh results folder when switching between a quick check and a full run. Existing checkpoints may resume; never interpret a resumed quick job as a full run without checking its recorded step count.
- If a dependency is missing, rerun the installation steps in `README.txt` using the same Python executable used for the command.
- If a checkpoint checksum fails, extract a fresh package copy. Do not edit a checksum to make changed parameters appear to be the supplied results.
- JAX compiles calculations on first use. CPU timings depend on machine and concurrent work; saved durations are elapsed run times, not processor-hours.
- A deliberate extreme-value NumPy test can emit an overflow warning while its clipping assertions pass. A non-zero exit code or failed assertion is a failure.
