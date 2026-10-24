=====================================================================================================
COMS4070A / COMS7081A  Computational Neuroscience - Project 2
Peer review + reproduction of
   Dorrell, Latham, Behrens & Whittington (2023), "Actionable Neural Representations: Grid Cells from
   Minimal Constraints", ICLR 2023 (arXiv:2209.15563)
=====================================================================================================

WHAT IS IN THIS FOLDER
----------------------
src/actionable/     the model (core.py), analysis tools (analysis.py), Appendix-K module-angle code
                    (module_angle.py) and plotting helpers.  Pure NumPy / SciPy / Matplotlib.
experiments/        one script per experiment + run_all.py, which runs everything.
tests/              unit tests (gradient checks against finite differences, actionability identity, gridness,
                    module clustering, lattice read-out, module-angle prediction, exact resume of training).
results/runs/       one folder per training run (config.json, params.npz, metrics.json, rate_maps.png, ...).
results/tables/     json tables behind every number quoted in the review.
results/figures/    figures made from the runs.
review/             the peer review (LaTeX sources, bibliography, compiled PDF), NeurIPS checklist, AI declaration.
requirements.txt    numpy, scipy, matplotlib.        README.txt  this file.

The code was written from the paper text only (no code from the authors was available to us).  Everything the
paper leaves unspecified is a documented, tested choice - see "CHOICES WE HAD TO MAKE" below and section
"Reproducibility" of the review.


1. INSTALL (any operating system)
---------------------------------
You need Python 3.9 or newer (we developed and tested with Python 3.13; numpy 2.5.3, scipy 1.18.1,
matplotlib 3.11.2 - other recent versions should work but we have not tested them).  No GPU is needed.

   Windows (PowerShell or cmd), in the project folder:
        py -m venv .venv
        .venv\Scripts\activate
        pip install -r requirements.txt

   macOS / Linux, in the project folder:
        python3 -m venv .venv
        source .venv/bin/activate
        pip install -r requirements.txt

   (Windows PowerShell may refuse to run activate; if so run once:
        Set-ExecutionPolicy -Scope CurrentUser RemoteSigned )

   Check the installation (takes a few seconds):
        python -m unittest discover -s tests -v


2. QUICK START
--------------
   a) 3-minute smoke test of the whole pipeline (tiny runs, results go to results_quick/):
        python experiments/run_all.py --quick

   b) Train ONE model like the paper's Fig. 2B (150 000 steps, about 12 minutes on one core), look at it:
        python experiments/exp1_grid_emergence.py --job main_sigma0.2_seed0
      -> results/runs/main_sigma0.2_seed0/rate_maps.png   (rate maps, frame colour = module)
         results/runs/main_sigma0.2_seed0/freq_space.png  (learned frequencies)
         results/runs/main_sigma0.2_seed0/metrics.json    (modules, lattice statistics, gridness, ...)

   c) Your own run (any setting of the model, see experiments/exp1_grid_emergence.py --help):
        python experiments/exp1_grid_emergence.py --name my_run --sigma 0.3 --N 64 --D 31 --steps 50000 --seed 7

   d) EVERYTHING (60 training runs + all analyses + figures + the LaTeX number macros):
        python experiments/run_all.py                # ~3.5 CPU-hours in total; default 2 runs in parallel
        python experiments/run_all.py --jobs 4       # use 4 cores
      Training is RESUMABLE: if the computer sleeps or you press Ctrl-C, start the same command again; every
      run continues from its last checkpoint (results/runs/<name>/ckpt.npz) and finished runs are skipped.
      Useful options:
        --only main oned            only these groups or job names (python experiments/jobs.py lists all of them)
        --reverse                   work through the runs from the END of the list: lets two computers share the
                                    work (one normal, one --reverse; copy results/runs/* from one to the other)
        --analyse-only              no training; re-analyse the finished runs and rebuild tables/figures/numbers

   e) Re-analyse without retraining (after editing src/actionable/analysis.py):
        python experiments/reanalyse.py

   f) Build the review PDF (needs a LaTeX installation, e.g. MiKTeX / TeX Live / MacTeX):
        cd review
        pdflatex main && bibtex main && pdflatex main && pdflatex main
      review/tables/numbers.tex (written by experiments/make_numbers.py) holds every number quoted in the text, so
      the text always agrees with the results you have just produced.

Timing (measured on a 2-core cloud machine; your PC will probably be faster): one 150k-step run 12 min, one 50k-step
run 4 min, all 60 runs about 3.5 CPU-hours.


3. WHICH SCRIPT PRODUCES WHAT
-----------------------------
exp1_grid_emergence.py   train + analyse one code (the workhorse).  Main results: runs main_sigma*_seed*.
exp2_1d_lattice.py       Section 3.1: 1-D codes with the simplified loss are lattices (harmonic frequencies).
exp3_sharpness.py        Section 4.1: N ~ (nu/mu)^2 (module size vs spacing/field-width), measured on all modules.
exp4_module_angle.py     Section 4.2 / Appendix K: predicted optimal angle between modules + angles in our modules.
exp5_ablations_sigma.py  Appendix I (ablations) and Appendix B.4 (effect of sigma and l; Figs. 8, 9).
exp6_sensitivity.py      our reproducibility audit: one-at-a-time changes of choices the paper does not specify.
make_figures.py          figures of the review.      make_numbers.py   LaTeX macros + tables for the review.
reanalyse.py             recompute metrics.json of every run from the saved parameters.
jobs.py                  the list of all runs (name, group, configuration).


4. THE MODEL IN ONE PARAGRAPH
-----------------------------
Actionable code (paper eq. 6):   g(x) = a0 + sum_{d=1..D} a_d sin(k_d . x) + b_d cos(k_d . x),   D < N/2,
with learned coefficients a0, a_d, b_d in R^N and learned 2-D wave vectors k_d.  Loss (eqs. 21-25): the functional
loss  mean_ij exp(-|g~_i - g~_j|^2 / 2 sigma^2) chi(x_i, x_j)  on M = 150 points drawn from p(x) = N(0, L^2 I),
plus GECO-weighted penalties for negative rates and for deviations from unit mean-square rate in S = 15 randomly
shifted copies of the room (the paper's way to emulate an infinite plane, Appendix B.1).  Adam, lr 0.1, beta1 =
beta2 = 0.9, 150 000 steps.  All gradients are written by hand (src/actionable/core.py) and verified by finite
differences (tests/test_core.py).


5. CHOICES WE HAD TO MAKE (the paper does not specify them)
-----------------------------------------------------------
 * number of frequencies D: 31 (largest integer < N/2 for N = 64);  one-at-a-time test with D = 20 (exp6)
 * initialisation: coefficients N(0, 1/D), frequencies N(0, 4^2);  tests with frequency scale 2 and 8 (exp6)
 * normalisation of the code (eq. 22): we read it literally - unit L2 norm over the M sampled points ("sum");
   the alternative, unit mean-square ("mean"), is tested in exp6
 * GECO target of the norm constraint k_n = +4 as in Table B.3 (with this value the multiplier decays to its floor,
   see the review); k_n = -4 is tested in exp6
 * Gaussian vs uniform occupancy for the 1-D runs; Adam epsilon 1e-8 (as given), float32 arithmetic
 * how to decide that a module is hexagonal: the paper gives no test.  We read the lattice off the module's
   autocorrelogram (six nearest peaks, angles within 15 degrees of a hexagon, distances within 25 %, analytic
   gridness > 0.3).  These thresholds are ours.


6. TROUBLESHOOTING
------------------
 * "No module named actionable": run the scripts from the project folder as shown (the scripts add src/ themselves).
 * Very slow on a laptop: set the environment variable OMP_NUM_THREADS=1 (run_all.py does this for you) and use
   --jobs equal to your number of physical cores.
 * Figures are not shown on screen: they are saved as png files (matplotlib "Agg" backend), nothing pops up.
 * You changed a model setting and get "checkpoint was produced with a different configuration":
   delete results/runs/<name>/ckpt.npz (or use a new --name).
