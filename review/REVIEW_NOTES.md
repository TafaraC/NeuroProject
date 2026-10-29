# Pre-submission review notes (branch `review-polish`)

Checked `review/paper_review/main.tex` against the full paper (incl. Appendices
B, G, H, I) and the brief/rubric. Main text fits within 3 pages after edits.

## Edits already made on this branch (factual / accuracy only)
1. **Summary** – "Predictions concern field width" → the paper's actual
   prediction is that the *grid spacing : field width* ratio scales with
   module size (Section 4.1, N ∝ (ν/µ)²).
2. **Reproducibility** – added that Figure 8 itself reports band-like/square
   grids for N=64 at l=0.1 (Appendix B.4.1: "at l = 0.1, where 64 neuron
   modules were not hexagonal, 100 neuron modules were"). Your l=0.1
   result therefore *agrees* with the paper; the real discrepancy is the
   l=0.5 baseline. This sharpens your reproduction finding.
3. **Significance** – the paper already uses Stensola et al. (2012)
   square→rectangle recordings (Fig. 5H–I) as support for the room-geometry
   prediction. The review now acknowledges this, so the proposed experiment
   reads as the quantitative follow-up rather than a new idea. (No 6th
   reference added – brief caps at 5.)
4. **Originality** – noted the Appendix I ablations are on a small torus, not
   the open plane (true per Appendix I; supports your "within this framework"
   point).
5. Grammar: "reinforce ... but does not refute" → "do not refute".
6. AI disclosure now also names Claude (needed for honesty, since this
   branch contains Claude edits).

## Suggestions for you to decide (opinions must be your own)
- **Balance vs. rubric weights.** Reproducibility is ~40% of the main text
  but 15% of the mark; Summary + Quality + Significance are 60%. Consider
  moving some numbers (norm errors, 3.61%/5.08%, window details) to the
  appendix captions and spending the space on Significance.
- **Module-orientation prediction (~4°, Section 4.2) is not discussed.** It
  is the paper's most distinctive, cheaply testable prediction (existing
  multi-module datasets could test it), and compared only "in a rudimentary
  analysis". A second, short theory–experiment loop here would strengthen
  Significance, and could also be a Quality point (no statistics on Fig 5E).
- **Harmonic tussle** (Section 3.3) is the paper's core explanation of
  modules; the Summary mentions modules but not why. One clause would help.
- **Jargon.** Brief penalises jargon-heavy reviews. Reproducibility
  paragraphs 4–5 ("spectral autocorrelation", "finite-window Pearson
  autocorrelation", "sampled norm penalty") could be simplified for a
  general reader.
- **Template.** The supplied `ccn_style.tex` has two `\phantom{.}\newpage`
  blank pages after the title; you omitted them. Probably fine, but worth a
  quick check with the lecturer if unsure.

## Still outstanding before upload
- Official Wits Science Faculty AI Declaration (`ai_declaration.tex` is a
  placeholder). It should mention both Codex and Claude.
- NeurIPS checklist: Q9 (ethics code) still says "pending author
  confirmation"; Q12/Q13 say licensing is unresolved. Decide and update.
- Rebuild the package (`python -m submission.build_package --output dist/<new>`)
  after the declaration is added; the old zips in `dist/` predate these edits.
- Could not independently re-run tests on a clean Linux install (no package
  index access from the checking environment); `submission/VALIDATION.md`
  records the Windows pass.
