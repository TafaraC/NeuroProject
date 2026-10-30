# Readable submission workspace

This folder holds a reorganised copy of the consolidated project. The original repository and the package in `dist/NeuroProject_combined_corrected/` remain unchanged.

| Location | Purpose |
|---|---|
| `NeuroProject_2683459_2717495/` | Submission package: report, runnable code, results and reader documentation |
| `editable_sources/review/` | Editable report source, course style and report figures |
| `build.py` | Copy source inputs and export the evidence tables |
| `file_inventory.csv` | Classification and disposition of every original package file |
| `preserved_files.json` | Internal byte-preservation check; excluded from the submission |
| `source_fingerprint.json` | Internal check that the original package was not changed |

## Requirements retained

The assignment explicitly requires the full implementation, `requirements.txt`, `README.txt`, template-based PDF review, NeurIPS checklist and Faculty AI declaration. It also requires a preliminary contribution statement. The review has six specified sections and a three-page limit excluding citations and supporting tables/figures. The brief recommends no more than five references. The supplied course style and both authors' names and student numbers remain in the report.

The optional NeuralPlayground integration is a separate contribution. No accepted upstream contribution is claimed by this package.

## Deliberate exceptions to the restructuring guide

- Executable code is unchanged. Therefore legacy internal scripts retain their original docstrings and command behaviour. The new `run.py` provides documented commands and help rather than modifying scientific code to add command-line interfaces.
- Configurations, checkpoint metadata, fixtures and snapshot manifests remain where the unchanged scripts read them. NumPy's number generator also reads training logs. Removing these would change or break reproduction, so readable CSVs supplement them.
- Snapshot-referenced plots remain inside the NumPy backing directory so its manifest has no missing targets. Other redundant plot galleries are omitted; the report's figures are kept with its editable sources.
- Original generated NumPy LaTeX tables are retained as script inputs. The editable review itself is outside the submission.
- The mandatory AI declarations will be supplied by the authors; no unsigned draft is substituted.

## Editing and rebuilding

1. Edit `editable_sources/review/main.tex`; keep its section order and the course template.
2. Compile that file with the built-in LaTeX compiler, Tectonic or two passes of `pdflatex` from its folder.
3. Copy its `main.pdf` to `NeuroProject_2683459_2717495/NeuroProject_Report.pdf`.
4. Verify byte-preserved inputs and evidence using `verify.py`, then build the ZIP using `package.py`.

`build.py` creates the initial copy and evidence tables. Re-running it copies the original report source again; preserve any edited source before doing so. It does not copy the final compiled PDF or replace the reader documentation. The scientific scripts in `code/review/` regenerate numerical macros in their original sibling layout; copy their generated `.tex` inputs to `editable_sources/review/` only after comparing the values.
