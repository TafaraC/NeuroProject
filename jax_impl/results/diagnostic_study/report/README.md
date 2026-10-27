# Baseline diagnostic study

Do sigma interpretation or the literal published norm target explain the baseline failure?

Final checkpoints only. Runs, not neurons, are replicates. No arm is an exact upstream rerun. A relaxed target passing is not biological success.

| Condition | Seed | Median grid score | Norm error | Mean negativity | Common targets met? |
| --- | --- | --- | --- | --- | --- |
| baseline | 0 | 0.0509 | 0.03011 | 9.9603e-05 | False |
| sigma_variance | 0 | 0.0666 | 0.024349 | 9.8344e-05 | False |
| norm_table | 0 | -0.1226 | 3.8727 | 0.00019954 | False |
| baseline | 1 | 0.0483 | 0.023593 | 0.00010196 | False |
| sigma_variance | 1 | 0.1006 | 0.014831 | 0.00012728 | False |
| norm_table | 1 | -0.0366 | 1.8786 | 0.00012162 | False |
| baseline | 2 | 0.0645 | 0.020855 | 9.235e-05 | False |
| sigma_variance | 2 | 0.0814 | 0.022185 | 0.00013726 | False |
| norm_table | 2 | -0.0627 | 1.7653 | 7.5517e-05 | False |

All planned final checkpoints are included. Same extent, resolution, evaluation seed and batches for every condition. Initial states were checked for exact equality within seed; training source and package versions were checked across runs.

Common assessment uses the ORIGINAL baseline norm tolerance, including for the relaxed norm arm. Neither tolerance satisfaction nor a positive grid score establishes a biological grid module. Functional objectives with different sigma are not directly compared. No p-values or neuron-level confidence intervals are reported.

The norm-table arm tests the literal log target in isolation; it does not also copy the upstream raw-rate norm formula or nested-log updates. Thus it is not a full upstream reproduction.

Changing sigma or the target changes the numerical problem; effects do not by themselves identify a bug or disprove the paper. Inspect all-cell maps and constraints before deciding whether the baseline is resolved.
