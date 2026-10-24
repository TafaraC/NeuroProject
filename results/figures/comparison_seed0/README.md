# Independent reproduction comparison

One paired training seed; descriptive results only.

| l | Median grid score | Positivity loss | Norm loss | Both held-out targets met? |
| --- | --- | --- | --- | --- |
| 0.5 | 0.0503 | 9.96026e-05 | 0.0301103 | False |
| 0.1 | 0.0088 | 0.000110589 | 0.0194958 | True |

These are final checkpoints, not hand-selected best cells or seeds. All-cell maps are available in each evaluation directory.
Grid scores maximise a rotational-correlation contrast over fixed annuli. They are not shuffled significance tests, and do not identify modules.
A positive score does not establish a biologically valid representation. Read the held-out constraint checks alongside the maps.
Repeated seeds, a direct module analysis, and biological validation remain outside this single-pair result.
The implementation follows the paper's normalised norm loss, with an explicit 0.02 tolerance; see references/implementation_notes.md.
