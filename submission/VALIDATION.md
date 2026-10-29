# Submission validation

Validated on Windows x64 with Python 3.14.3 and the pinned scientific runtime.
The extracted package was tested using the existing verified environment;
a second clean dependency installation was not performed.

- Tafara: 15 pytest tests passed from an extracted archive (20.36 seconds).
- Nathan: 20 unittest tests passed from the extracted snapshot (1.01 seconds).
- Nathan's deliberate GECO extreme-value test emits an overflow warning and
  passes its clipping assertions; this is not a failed training run.
- The shared equation audit passes from local packaged source without Git.
  Packaging explicitly disables Git's line-ending conversion to preserve the
  inspected source digest.
- The snapshot comparison reads all 60 completed checkpoint folders, checks
  their saved bytes, and recognises the five existing common evaluations.
  These five full evaluations were performed on 26 October; packaging does not
  retrain them or introduce new results.
- Nathan's direct experiment CLI completed a separate 1,500-step quick run.
  This is a workflow check and is excluded from the scientific evidence.
- Pinned dependency consistency check and Ruff checks passed.
- Both LaTeX sources compile. The joint review has seven pages: main review
  within pages 1--3, supporting evidence on pages 4--6, checklist on page 7.
  All pages and the separate contribution statement were visually checked.
- The archive includes a complete file manifest and a second manifest for the
  pinned Nathan snapshot. ZIP integrity and file digests are checked before
  delivery.

The first test attempt could not create temporary folders in the restricted
execution environment. Repeating with a writable project-local temporary
directory resolved the setup errors; no model code was changed for this.

Remaining author action: append the completed official AI Declaration through
`review/paper_review/ai_declaration.tex`, recompile and replace root Review.pdf,
then refresh the file manifest and zip the final folder for upload.
