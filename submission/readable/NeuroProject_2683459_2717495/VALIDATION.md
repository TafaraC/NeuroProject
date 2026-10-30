# Checks performed on this package

This package reorganises the project for reading. It does not change any code, configuration, checkpoint or recorded result. The checks below confirm that.

| Check | How it was done | Result |
|---|---|---|
| Code and saved inputs unchanged | Every copied file was hashed (SHA-256) and compared with the original project files. | 759 copied files are byte-identical. The 862 original files were not modified. |
| NumPy snapshot complete | Each file listed in `code/numpy_impl/SNAPSHOT.json` was hashed and compared. | All listed files are present and identical. |
| Evidence tables match the saved results | Each row in `evidence/` was compared with the JSON or CSV file named in its `original_run` column. | All values match exactly. No value was rounded. |
| Headline outcomes | The module counts in `evidence/numpy_runs.csv` were checked against the saved metrics. | Both full σ = 1.0 runs have one hexagonal module. All three σ = 0.2 runs have none. |
| Report numbers unchanged | Every numeral in the report source was compared with the earlier report. The section order, labels and figures were also compared. | All are unchanged. Only the wording was edited. |
| Report builds | The report was compiled three times with `pdflatex`, and every page was inspected. | No errors and no overfull lines. 9 pages in total; the main review fits in 3 pages. |
| NumPy tests | Ran `python -m unittest discover -s tests` in `code/numpy_impl/` (Linux, Python 3.13, NumPy 2.5, SciPy 1.18). | All 20 tests pass. One overflow warning comes from a deliberate extreme-value test. |
| Both test suites on Windows | Ran `python run.py tests` from this folder with the project's Windows environment (Python 3.14, JAX 0.11.2 on CPU). | All 20 NumPy tests and all 15 JAX tests pass. |
| Wording | Searched the package for "deadline", "lecturer", "TBD", "placeholder" and process terms. Also searched for files written for AI agents. | None remain in text a reader sees. AI use is disclosed in `CONTRIBUTIONS.md` and the declarations. |
| File references | Checked that every file named in a README exists. | All exist. The `reproduced_results/` subfolders are created when you run the commands. |

The package has not yet been tested by someone new to the project. That person should find a result using `evidence/README.md`, run `python run.py tests`, and follow `code/RUN_GUIDE.md`.
