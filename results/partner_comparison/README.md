# Independent implementation comparison

Update, 26 October: Nathan's newer results at `df44d69` are compared in
`../partner_results_20261026/README.md`, including a complete checkpoint inventory
and five evaluations using Tafara's common settings. The core source is unchanged,
so the equation audit below still applies. Its earlier comments on the number of
available completed runs describe the 25 October snapshot only.

Compared Tafara's JAX equations with Nathan's NumPy implementation at commit
`7db3147a434adb20b9b0514e3201bec40700d99b` (`origin/Nathan`, fetched 25 October
2026). The model implementations remain separate. No partner training code was
merged into `src/`, and no partner runs were resumed or modified.

## Measured agreement

The shared float64 fixture uses identical coefficients, frequencies, sampled
positions, room shifts, and perturbation directions. Nathan's unmodified core
agrees with the independently calculated NumPy reference on the three raw loss
terms, with maximum absolute difference **2.77e-13**. The JAX implementation
passes the same reference test.

There is an intentional objective-weight difference: our functional term has a
factor 25, whereas Nathan's has factor 1. To compare derivatives of the SAME
objective, the adapter divides Nathan's constraint weights by 25 before calling
his core, then multiplies its gradients by 25. After this explicit alignment,
the maximum difference from the finite-difference directional derivatives is
**5.09e-11**. The total objectives with the unmodified multipliers are different
(0.174861 versus 2.335752 on this fixture), as expected.

All six comparisons pass the predeclared rtol=2e-5 / atol=1e-7 thresholds.
`comparison.json` contains every value, source/fixture hashes and interpretation.
This is strong evidence of equation agreement on this input, not a proof of
equivalence of all training code or successful reproduction of the paper.

## Why the original training runs cannot be compared directly

| Choice | Tafara's original baseline | Nathan's saved main run |
| --- | --- | --- |
| Main target | Figure 8, sigma=0.4 | Figure 2B, sigma=0.2 |
| Functional term multiplier | 25 | 1 |
| Norm target | 0.02, explicit choice | exp(4), literal table target |
| Frequency initialisation | Uniform [0,2) in each axis | Zero-mean Gaussian, standard deviation 4 |
| Coefficient initialisation | Standard normal | Standard normal divided by sqrt(D) |
| Adam stabiliser | sqrt(v + 1e-8) | sqrt(v) + 1e-8 |
| Random generator | JAX key | NumPy generator |
| Grid analysis | Finite-window Pearson SAC; fixed physical annuli | Empirical and analytic SAC; module/spacing-dependent analysis |

Both use sampled-sum original-room normalisation and normalised equation-25
shifted-room norm errors. Both interpret sigma as a standard deviation. A given
integer seed does not imply identical initial arrays between JAX and NumPy.
Their GECO clipping and tiny-loss floors also differ, so matching target names
alone does not align all optimiser behaviour.

Nathan's branch includes two completed sigma=0.2 runs and additional partial
checkpoints; this audit did not re-evaluate their spatial metrics. It does not
establish which implementation gives a better result. Our diagnostic study was
specified and launched before inspecting Nathan's branch and remains unchanged.

## Repeat the audit

With the pinned commit available locally (fetch `origin/Nathan` if necessary):

```powershell
.\.venv\Scripts\python.exe -m comparison.partner_equations --output results/data/partner_equation_recheck
.\.venv\Scripts\python.exe -m experiments.equation_check --output experiments/fixtures/equation_contract --candidate results/data/partner_equation_recheck/partner_candidate.json
```

The adapter only executes the inspected source hash at the pinned commit. It
loads the numerical core from Git without checking out or merging the branch.
The general fixture checker remains available for further implementations.

Next, align training conventions deliberately before a matched-optimiser
comparison, and use the same evaluation on both final populations. Keep these
equation-level results distinct from claims about emergent grid geometry.
