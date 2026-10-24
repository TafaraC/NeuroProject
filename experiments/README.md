# Reproduction experiments

Scope: one baseline reproduction and one small comparison for a two-person,
two-week project. The model, training, evaluation, and comparison entry points
are implemented; commands are in the repository README.

1. Reproduce a single module of hexagonal grids (Figure 4B; Appendix B).
2. Once the baseline works, compare two spatial discrimination scales with a
   fixed neuron population, following Figure 8 / Appendix B.4.
3. Assess the baseline across matched seeds before interpreting a spatial-scale effect.

The original pair did not establish the baseline. The current follow-up is the
controlled three-condition, three-seed study in `diagnostics.md`, which tests
two documented numerical ambiguities separately. `equation_check.py` provides
a shared numerical contract for comparing Tafara's and Nathan's implementations.

Proposed comparison: N = 64, sigma = 0.4, l = 0.5 versus l = 0.1.
Frequency count: 31. Configuration defaults and every resolved run setting are
saved to JSON. See `references/implementation_notes.md` for the equation audit,
normalisation conventions, constraint targets, and reference-code differences.

Each run records its configuration, seed, package versions, source hashes, code
revision, and losses. Evaluation measures constraint violations, rate maps,
spatial autocorrelations, and descriptive grid scores. Unsuccessful runs are
retained as evidence. Final checkpoints, rather than selected best cells, are compared.

Each run is self-contained under results/data: config, metadata, history,
initial and resumable checkpoints, status, and evaluations. results/figures
contains the compact comparison exports suitable for version control. The
older empty checkpoints/logs directories remain available, but the runner uses
per-run folders to avoid separating parameters from their configuration.
