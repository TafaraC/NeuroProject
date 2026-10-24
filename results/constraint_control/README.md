# A constructed hexagonal control for the sampled norm penalty

This is an **exploratory diagnostic control, not a trained model or successful
reproduction**. It was added after noticing the sensitivity to the norm target.
It does not change the already specified nine-run training study.

Each of 64 neurons is a translated copy of

g(x) = 1.5 + cos(u) + cos(v) + cos(u+v)
     = 0.5 * |1 + exp(iu) + exp(-iv)|^2.

The three wavevectors have equal magnitude 6 and lie at 0, 60 and 120 degrees.
The squared-magnitude identity proves non-negativity; the code is bounded above
by 4.5. There are only three frequency pairs, so this control is not the trained
31-frequency population used in the reproduction study.

We measure equation 25 using the same original-room sampled-sum normalisation,
Gaussian occupancy (standard deviation 1), and 15 Gaussian room shifts (standard
deviation 3). The 128 batches are independent sampling draws for a FIXED code;
they are not independent training seeds. Sample counts are paired using nested
position arrays and the same shifts within each batch.

| Sample count M | Mean sampled norm loss | Monte Carlo standard error | Batches meeting 0.02 |
| --- | --- | --- | --- |
| 50 | 0.09610 | 0.00575 | 3.91% |
| 150 | 0.02738 | 0.00129 | 34.38% |
| 600 | 0.00644 | 0.00029 | 100% |

Exact Gaussian integration of the finite Fourier expansion gives a maximum
relative shifted-room energy deviation of only **7.16e-8** over the tested rooms.
That calculation uses the Gaussian characteristic function, independently checked
against Gaussian quadrature in tests. Thus the substantially larger sampled norm
penalty is dominated here by finite-sample estimation, despite valid hexagonal
geometry and exact non-negativity.

This does not prove why training failed or establish a universally correct norm
target. It shows that our chosen 0.02 target at M=150 can reject a known valid
hexagonal code on the sampled criterion. Target calibration and sampling precision
must be distinguished from biological validity. Increasing M also changes neural
distance scale under SUM normalisation, so a training experiment varying M would
need to account for sigma; this control measures the norm penalty only.

All input parameters, the NumPy version, source hash and measurements are in
`summary.json`; per-batch scores are in `batches.csv`. The constructed coefficients
are in `constructed_code.npz`. The first figure panel is explicitly labelled as a
constructed control, never an emergent training result.

Repeat from the repository root with a new output directory:

```powershell
.\.venv\Scripts\python.exe -m comparison.constraint_control --output results/data/constraint_control_recheck
```
