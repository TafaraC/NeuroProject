# Before you submit (due Tuesday 27 October 2026, 17:00 SAST)

Tick each box yourself; the drafting assistant cannot do these for you.

## Must do
- [ ] **Read `review/main.pdf` slowly and make it yours.** The brief says the opinions must be yours and that clear
      hallucinations score 0. Judgements to confirm or change (all in the six sections):
      periodic tuning is "largely a consequence of the formulation" (Originality); the abstraction level is right for a
      "why" question but weak for biological claims (Originality/Quality); "several hexagonal modules" is the weakest
      part of the central result (Quality); the hexagonal outcome "seems to depend on how loosely boundedness is
      enforced" (Reproducibility, flagged there as a hypothesis); the three proposed experiments (Significance).
- [ ] **Verify the two statements about the authors' public code** (end of Clarity; also implied in Reproducibility):
      the norm routine aggregates unnormalised responses where equation 25 uses normalised ones, and `sigma_sq` must be
      distinguished from the paper's sigma. They come from Tafara's inspection of commit 940659e and were not checked by
      the assistant. If you cannot confirm them, delete that paragraph (the review still stands).
- [ ] **Attach the official Wits Science Faculty AI declaration**, completed and signed by both authors
      (`declarations/AI_declaration_notes.txt` is only a worksheet).
- [ ] **Check the course template.** `review/ccn.sty` is the style file Tafara used (the README of his repository says
      it is the unmodified file supplied with the course). Compare with the Moodle template and port anything else it
      requires (title block, header, line numbers).
- [ ] **Check the contribution wording** ("Author contributions and AI use") against the preliminary contribution
      statement you submitted two weeks before the deadline.
- [ ] **Confirm checklist answers 9 and 12** (code of ethics; licences) and the student numbers on the title.
- [ ] **Run `python smoke_test.py` on your own machine** (about 2 minutes) after unzipping.

## Check
- [ ] The body of the review is at most 3 pages (it ends on page 3, before the references).
- [ ] At most 5 references (there are exactly 5; the reviewed paper is cited in the text, not listed).
- [ ] The zip size is acceptable to Moodle (about 120 MB). If it is too large, delete the per-neuron picture galleries in
      `jax_impl/results/diagnostic_study/report/*/` (about 50 MB; they can be regenerated) and mention it in README.txt.
- [ ] If you re-run anything: `python review/make_review_numbers.py`, rebuild the PDF twice, re-check the page count and
      re-read every sentence whose numbers changed.

## Open choices you may want to revisit
- Reference list: Gao 2021, Sorscher 2019, Burak & Fiete 2009, Stensola 2012, Gardner 2022 (swap freely, keep 5).
- Two extra NumPy jobs from your earlier plan (N = 128 neurons at sigma 0.2) and the 1-D ring experiment are not in the
  zips this package was built from. If you have them, drop them into `numpy_impl/` and re-run
  `python review/make_review_numbers.py`; the review text would then need a sentence each.
- The optional NeuralPlayground contribution (due 27 November 2026) is not started.
