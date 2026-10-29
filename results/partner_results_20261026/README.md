# Tafara and Nathan: comparison on 26 October 2026

Nathan's latest fetched revision is `df44d694e56ed91d5c4f5ea7499d1ca85d0bf046`.
It adds three commits after the earlier equation audit: `ef92dbb`, `4fc207b`,
and `df44d69`. Tafara's comparison point is `a6e5ae9`.
The implementations remain on separate branches; nothing was merged into the
training model. Nathan's tracked source and results were downloaded through Git.

## What changed

Nathan now has 60 saved experiment folders whose checkpoint step counts reach
their configured budgets. They contain 59 distinct parameter sets:
`sweepl02_sigma0.4_seed0` and `sweepl_l0.2_sigma0.4` are duplicates and must not
be counted as independent repetitions. Budgets vary from 30k to 150k steps.
`inventory.csv` lists every folder, budget, completed step and reported metric.
His new work includes sigma/spatial-scale/population-size sweeps, ablations,
one-dimensional runs, module analysis and sensitivity experiments.

Neither `src/actionable/core.py` nor `src/actionable/analysis.py` changed since
the pinned `7db3147` audit. The earlier small-input equation agreement therefore
still applies to the current core. It does not establish training equivalence.
His root README is stale: it still says no model is implemented, despite the
saved code and results. Use actual configs/checkpoints as the evidence.

## Nathan's reported results

| Runs | Budget each | Reported hexagonal modules | Fraction of cells with analytic grid score >0.3 |
| --- | --- | --- | --- |
| sigma=1.0, seeds 0 and 1 | 150k | 1 in each | 100% in each |
| sigma=0.2, seeds 0, 1 and 2 | 150k | 0 in each | 14.1%, 0%, 0% |
| sigma=0.4, seeds 0 and 1 | 50k | 1, 0 | 100%, 0% |

These are Nathan's descriptive classifications, not shuffle-calibrated
significance tests. His hexagonal-module criterion combines six-peak angle
error <15 degrees, radius spread <25%, and analytic grid score >0.3.
The sigma=1.0 modules have angle errors of 2.14 and 1.20 degrees and radius
spreads of 3.38% and 1.96%. This is credible evidence of hexagonal geometry
under those settings, stronger than Tafara's elongated baseline maps.
It does not establish robustness at the smaller sigma used in his main target.

His tighter-norm sigma=1.0 sensitivity run (50k, log target -4) reports no
hexagonal module, unlike his sigma=1.0 baseline's 50k snapshot. Together with
Tafara's diagnostic/control findings, this makes constraint calibration a
particularly useful shared research question. It remains a sensitivity result,
not proof that the constraint alone explains every difference.

## Why this is not a ranking of implementations

Tafara's original runs use sigma=0.4, a functional multiplier of 25, norm target
0.02, uniform initial frequencies and Adam epsilon inside the square root.
Nathan uses functional multiplier 1, norm target exp(4), Gaussian initial
frequencies and epsilon outside the square root. Coefficient scales and random
generators also differ. An integer seed does not create matched initial arrays.
Nathan's reported training losses are averages of the last five logged values;
Tafara's reported constraint losses are held-out evaluations.

Nathan's analytic autocorrelation uses each cell's spectrum and a window scaled
to its spatial period. Tafara's score uses a finite rate-map window and fixed
annuli. Raw grid scores and fractions from the two analyses are not comparable.
In particular, a fixed small window may contain too few repeats of a broad grid.
The common-window check below complements, rather than overrides, the analytic
geometry evidence.

## Common evaluator

`comparison/partner_results.py` reads saved arrays directly from an immutable Git
revision without executing Nathan's code or retraining. It verifies the sine/cosine
basis conversion against independent response calculations. Five runs are
selected explicitly: the sigma=0.2 seed-0 main run, both sigma=1.0 main runs,
and both sigma=0.4 sweep runs. This is a targeted comparison, not a re-evaluation
of all 60 folders.

Each receives the same 352-by-352 grid over [-1.5,1.5]^2, eight held-out batches,
seed 12345, sum normalisation and common norm tolerance 0.02 used for Tafara's
nine-run diagnostic report. `common_evaluation.csv` and per-run summaries give
these measurements. The summaries retain Nathan's actual training configuration
separately: the evaluator's `config` must not be mistaken for his optimiser.
All other retained metadata is descriptive; no significance or module-count
claim is inferred from these finite-window scores.

| Nathan checkpoint | Common median grid score | Held-out norm loss | Held-out mean negativity |
| --- | --- | --- | --- |
| sigma=0.2, seed 0, 150k | 0.0238 | 5.8261 | 0.0001878 |
| sigma=1.0, seed 0, 150k | -0.0969 | 3.1925 | 0.0001943 |
| sigma=1.0, seed 1, 150k | -0.1907 | 2.6302 | 0.0000858 |
| sigma=0.4, seed 0, 50k | -0.0809 | 2.5480 | 0.0000770 |
| sigma=0.4, seed 1, 50k | -0.0414 | 1.8502 | 0.0001619 |

For comparison, Tafara's three original-setting baseline runs (150k each) have
common median scores 0.0483--0.0645 and norm losses 0.02086--0.03011. Tafara's
relaxed-target runs have norm losses 1.7653--3.8727, broadly comparable in scale
to Nathan's large errors, though the training setups still differ. All five
re-evaluated Nathan checkpoints exceed the common 0.02 norm threshold. That
threshold is our choice and is subject to the sampling caveat established in
the constructed-code control; exceeding it alone does not invalidate geometry.

The negative common-window scores for Nathan's broad grids show that the
analysis window materially affects the apparent outcome. They should not be
interpreted as disproving his analytic hexagonality measurements or as a reason
to rank Tafara's narrow-period patterns higher. Both analyses should be retained
and their spatial scales explained in a joint report.

Repeat from the repository root after fetching Nathan's branch:

```powershell
.\.venv\Scripts\python.exe -m comparison.partner_results --revision df44d694e56ed91d5c4f5ea7499d1ca85d0bf046 --output results/data/partner_comparison_repeat
```

## Implication for the joint project

The combined account should report both the settings where Nathan finds
hexagonal geometry and the settings where either implementation does not.
Tafara contributes the independent equation checks, controlled three-seed
diagnostics and constructed-code sampling calibration. Nathan contributes a
broader parameter study and positive geometry examples. A fair causal follow-up
would exchange identical starting arrays and samples, align the objective and
optimiser conventions, and use both common-window and scale-aware analysis.
Neither the positive examples nor the negative runs alone justify a universal
claim about reproducibility. The existing review describes Tafara's results;
incorporating Nathan's new findings would change the joint evidence section.
