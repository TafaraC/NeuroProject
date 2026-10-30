# Independent implementation specification

Target: single-module hexagonal grids and the l sensitivity comparison in
Dorrell et al. (ICLR 2023), Figure 4B and Figure 8 / Appendix B.4.1.
This branch is an independent implementation for later comparison with the
groupmate's work. No code or results from that work are incorporated.

## Model and objective

For N=64 neurons, D=floor((N-1)/2)=31 two-dimensional frequencies, represent
position using g(x)=W[1, cos(k1.x), sin(k1.x), ..., cos(kD.x), sin(kD.x)].
Both W and k are learned; there is no pre-imposed lattice or rectifying nonlinearity.
The 63-column weight matrix must retain full column rank for exact actionability;
evaluation reports its rank, conditioning, and the translation residual using
T(delta)=W R(delta) pinv(W) with a fixed row normalisation.

Following Appendix B.1 equations 22--25, each neuron is normalised by the square
root of the SUM of its squared rates on M=150 original-room points. We use this
same normaliser in every shifted room. Points come from N(0,I); 15 shifts come
from N(0,9I); points and shifts are resampled every five steps.

- Functional loss: mean over all M^2 pairs of
  exp(-||g_i-g_j||^2/(2*sigma^2)) * [1-exp(-||x_i-x_j||^2/(2*l^2))].
  Diagonal contributions are zero. sigma is a STANDARD DEVIATION, 0.4.
- Positivity loss: mean negative part of normalised firing over shifted rooms.
- Norm loss: mean over rooms and neurons of (sum_over_points(g^2)-1)^2.
- Optimised total: 25*functional + lambda_pos*positivity + lambda_norm*norm.

The sampled-sum convention is deliberate: changing to RMS normalisation changes
the neural distance scale by sqrt(M), so sigma=0.4 no longer has the same meaning.
The main-text integral is an expectation, whereas equation 22 writes a sampled
sum; this implementation follows the appendix's numerical convention and keeps
M fixed across the comparison. It is not claimed to be invariant to batch size.

## Numerical choices and departures

The authors' public code was inspected at commit
`940659ed2773f4cbb3bcf298bdb70497b30d250c`:

- https://github.com/WilburDoz/ICLR_Actionable_Reps/blob/940659ed2773f4cbb3bcf298bdb70497b30d250c/Plane.py
- https://github.com/WilburDoz/ICLR_Actionable_Reps/blob/940659ed2773f4cbb3bcf298bdb70497b30d250c/NRT_functions/losses.py
- Paper: https://arxiv.org/abs/2209.15563

This code was written afresh from the equations; the reference informed the
implementation audit and initialisation/optimiser choices. It is not an exact
re-execution of the public script.

1. The public `norm_plane` calculates a normalised response `g`, then aggregates
   `g_other` (unnormalised), and uses a different aggregation from equation 25.
   Our norm loss uses the normalised response and squared deviations in equation
   25. Unit tests explicitly catch renormalisation of each shifted room and
   dependence on arbitrary per-neuron gains.
2. The paper's table gives k_norm=4, and the public script uses 6, but these do not
   specify a tight tolerance for the equation-25 loss under log(L)-k updates.
   We explicitly choose norm_target=0.02 (log target approximately -3.912).
   This is a REPRODUCTION CHOICE, not a parameter reported for Figure 8.
   It is an average squared room-energy error, not an absolute activity bound
   and not a guarantee that every neuron/room passes. Held-out p95 errors are
   reported. Positivity_target=exp(-9) follows the paper.
3. The public loss takes `sigma_sq` directly as the variance. Our sigma=0.4
   means variance=0.16, following the paper's notation. These must not be confused.
4. The public functional loss divides by `phi.size**2` for M two-dimensional
   points, and its caller multiplies by 100. Relative to a pairwise mean, this
   gives an effective factor 25. We make that factor explicit in the total loss.
5. W starts as independent N(0,1) entries; frequencies start uniform on [0,2)^2,
   following the public script's random scheme. The appendix says Gaussian
   initialisation but does not give the claimed variances in table B.3. No grid
   geometry is built into initial frequencies. A matched seed shares initial
   weights/frequencies and the random sampling stream across the two l values.
6. Adam uses beta1=beta2=0.9, learning rates 0.1, and epsilon=1e-8 INSIDE the
   square root, matching the public update. We implement this directly to avoid
   silently using an optimiser library's different epsilon convention.
7. GECO follows equations 26--28: smoothed log(loss/target), rate 1e-4, smoothing
   0.9. Updates affect the NEXT parameter step. We do not use the public script's
   extra nested log for positivity. Multipliers start at 0.1 and 0.005 and are
   represented in log space, clipped to [-20,20] to avoid under/overflow. A hit
   on the clipping boundary is exposed in the evaluation report.
8. Training uses float32. Equation, gradient and resume tests use float64.
   Resampling and initialisation use one saved JAX PRNG stream, unlike the
   reference script's mix of seeded JAX and unseeded NumPy sampling.

These choices can materially affect convergence. The result tests a clearly
specified interpretation of the model, not exact pixel-level agreement with a
published figure. Do not conceal the norm-tolerance or sigma differences when
comparing implementations.

## Evaluation and interpretation

Final checkpoints are evaluated using eight independent held-out batches from
the same distributions. Both l conditions use the same evaluation seed and
map grid (320 by 320, [-1.5,1.5] in each coordinate). A guard requires at least
four samples per shortest axis wavelength across ALL learned frequencies;
insufficient resolution is an error rather than a silently aliased figure.
This does not establish positivity
on the infinite plane. All 64 neurons are plotted in index order; figures are
not selected by their grid score. Initial checkpoints are retained for comparison.

Spatial autocorrelation is overlap-normalised Pearson correlation, with overlaps
below 20% of the map omitted. Rotational grid score is
min(r60,r120)-max(r30,r90,r150), maximised over 12 fixed annuli whose outer radii
range from 0.15 to 0.70 times the map width, with inner radius half the outer.
Rotations use bilinear interpolation and identical valid pixels for all angles.
Constant maps receive an undefined score. Annulus optimisation can inflate scores;
scores are descriptive, not significance tests. No independent-neuron error bars,
module count, or biological success is inferred from them.

The first complete comparison uses one seed per condition. Additional seeds are
needed for conclusions about robustness. Check constraint violations as well as
maps: a visually hexagonal result with unmet constraints is only a partial
reproduction. Full source hashes, package versions, resolved settings, random
state, optimiser moments, GECO state, and sampled rooms are recorded for audit
and exact continuation within the same software environment.

The first evaluation used 96 points across [-5,5]. Visual inspection and the
learned spectra showed that this undersampled the final frequencies. Those
evaluations are marked SUPERSEDED locally; final figures and scores are regenerated
on the shared finer grid above. Training and the held-out constraints were not
changed. Changing the plot extent changes the physical annuli used for scoring;
only scores on the same grid should be compared. This correction is part of the
evaluation audit, not a retraining or selection of a better checkpoint.
