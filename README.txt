COMS4070A / COMS7081A Computational Neuroscience, Project 2 (University of the Witwatersrand)
Peer review and reproduction of
  Dorrell, Latham, Behrens & Whittington (2023), "Actionable Neural Representations: Grid Cells
  from Minimal Constraints", ICLR 2023 (arXiv:2209.15563).
Authors: Tafara Chibanda (2683459), Nathan Le Roux (2717495)

=======================================================================================================
1. WHAT IS IN THIS FOLDER
=======================================================================================================
  review/              the peer review. review/main.pdf (source: main.tex) holds the six rubric sections
                       (3 pages of text), references, the appendix, and the NeurIPS checklist.
  numpy_impl/          implementation 1: NumPy, hand-derived gradients, written from the paper's text only.
                       60 saved training runs.  Own README: numpy_impl/README.txt
  jax_impl/            implementation 2: JAX, automatic differentiation.  Saved reference runs, a 9-run
                       diagnostic study, a constructed-code control and an equation audit.
                       Own READMEs: jax_impl/README.txt, jax_impl/README.md
  smoke_test.py        two-minute installation and sanity check of everything (see section 2)
  requirements.txt     packages needed (numpy, scipy, matplotlib, jax, jaxlib, pytest)
  requirements-pinned.txt   exact versions used while developing the JAX implementation (optional)
  declarations/        notes for the Wits Science Faculty AI declaration and a pre-submission checklist.
                       The official AI declaration form itself is NOT part of this folder.

No GPU and no PyTorch are needed. Everything runs on a CPU in plain Python 3 (3.12 tested on Linux;
the NumPy implementation was also run on Windows with Python 3.12/3.13, the JAX implementation on
Windows with Python 3.14).

=======================================================================================================
2. INSTALL AND CHECK  (about 2 minutes)
=======================================================================================================
Run these in this folder.

  Windows (PowerShell)                      macOS / Linux
  --------------------------------------    ------------------------------------
  py -m venv .venv                          python3 -m venv .venv
  .venv\Scripts\activate                    source .venv/bin/activate
  pip install -r requirements.txt           pip install -r requirements.txt
  python smoke_test.py                      python smoke_test.py

smoke_test.py runs (1) the NumPy unit tests, (2) a 300-step NumPy training run with analysis, (3) the JAX
unit tests, (4) a 2000-step JAX training run with evaluation, and (5) the shared-input comparison of the two
implementations' loss and gradient code. It writes only to a temporary folder and prints PASS/FAIL per step.
"python smoke_test.py --skip-jax" checks the NumPy implementation alone.
The short runs only check the installation; they are not reproductions of the paper's figures.

=======================================================================================================
3. RE-RUNNING THE EXPERIMENTS
=======================================================================================================
All commands are run from the folder of the implementation. Finished runs are never overwritten.

NumPy implementation (cd numpy_impl):
  python -m unittest discover -s tests -v      # 20 tests, about 2 s
  python experiments/run_all.py --quick        # 3-minute smoke test of the whole pipeline (writes results_quick/)
  python experiments/run_all.py --jobs 4       # all 60 runs + analyses, tables, figures, number macros
                                               # (about 5 CPU-hours; we needed roughly 70-80 minutes with 4 processes)
  python experiments/run_all.py --analyse-only # recompute metrics, tables, figures and numbers from saved runs
  python experiments/exp1_grid_emergence.py --job main_sigma0.2_seed0    # one run (150000 steps, about 15 min)
  Training resumes after an interruption: start the same command again.

JAX implementation (cd jax_impl):
  python -m pytest tests -q                    # 15 tests, about 30 s
  python -m experiments.run --config experiments/configs/baseline.json --output results/data/baseline_seed0 --evaluate
                                               # 150000 steps, about 10 min; use a NEW output folder each time
  python -m experiments.diagnostics plan|run|report --output results/data/my_diagnostics   # 9-run study
  See jax_impl/README.txt for the complete list of commands.

Cross-check of the two implementations (cd jax_impl):
  python -m comparison.partner_equations --output results/data/partner_equation_recheck
  Loads numpy_impl/src/actionable/core.py (its SHA-256 is verified) and compares its losses and
  directional derivatives with the JAX equation contract on shared float64 inputs.
  Stored result: jax_impl/results/partner_comparison (loss terms agree to 2.8e-13, derivatives to 5.1e-11).

=======================================================================================================
4. THE REVIEW
=======================================================================================================
Every number quoted in review/main.tex is a LaTeX macro generated from the saved result files:
  python review/make_review_numbers.py        # reads numpy_impl/results and jax_impl/results
  (add --regenerate to re-run numpy_impl/experiments/make_numbers.py first)
  python review/make_figures.py               # redraws review/figures/norm_target_montage.png (needs jax)
Build the PDF (needs a LaTeX installation such as TeX Live, MiKTeX or MacTeX), from review/:
  pdflatex main && pdflatex main
The review/ folder is self-contained, so it can also be uploaded to Overleaf as it is.
review/ccn.sty is the course style file, used unmodified.

=======================================================================================================
5. HOW THE TWO IMPLEMENTATIONS DIFFER (important for reading the reproducibility results)
=======================================================================================================
  * NumPy implementation: written from the paper's text only. The authors' public code (linked in
    Appendix B of the paper) was deliberately NOT read. Choices where the paper is silent are listed in
    numpy_impl/README.txt and varied one at a time in numpy_impl/experiments/exp6_sensitivity.py.
  * JAX implementation: the authors' public code (commit 940659e) was inspected to audit conventions and to
    choose initialisation and optimiser settings; no code was copied. Differences found between the paper
    and that code, and every other choice, are in jax_impl/references/implementation_notes.md.
  * Both implementations use the same equations (see the cross-check above) but different training
    conventions (functional weight, norm target, initialisation, Adam epsilon), so their runs are not
    interchangeable; the review compares them explicitly.
  * Training is chaotic: another machine gives similar but not identical runs. The saved runs are the
    ones the review reports.

=======================================================================================================
6. AI USE
=======================================================================================================
Claude (Anthropic) wrote the NumPy implementation, its tests and analysis scripts, and drafted the merged
review. OpenAI Codex assisted with the JAX implementation, its tests, analysis and an earlier review draft.
See "Author contributions and AI use" in the review and declarations/AI_declaration_notes.txt.
