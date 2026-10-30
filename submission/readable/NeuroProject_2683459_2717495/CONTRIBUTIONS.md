# Contributions and resources

| Member | Student number | Focus |
|---|---|---|
| Tafara Chibanda | 2683459 | JAX reproduction, matched-seed diagnostics, equation comparison, sampling-calibration control and initial review |
| Nathan Le Roux | 2717495 | NumPy reproduction, parameter sweeps, ablations and module analysis |

Both authors reviewed and agreed with the review's judgements and confirmed their contributions.

## Use of generative AI

Both authors used generative AI tools substantially. The signed Faculty declarations in `declarations/` list the same tools and uses; this table gives more detail.

| Use | Tool | What it did |
|---|---|---|
| Code generation | Claude (Anthropic) | Wrote the NumPy model, hand-derived gradients, tests and analysis scripts. |
| Code generation | OpenAI Codex | Wrote the JAX model, training loop, tests, diagnostics and comparison scripts. |
| Experiment automation | Claude (Anthropic) | Ran the NumPy parameter sweeps, ablations and module analysis. |
| Experiment automation | OpenAI Codex | Ran, checkpointed and evaluated the JAX reference and diagnostic training runs. |
| Content summary and writing | Claude and Codex | Summarised the paper, results and each other's code; drafted and edited the review, tables and figures. |
| Checking and packaging | Claude and Codex | Checked the review against the paper and saved results; organised and verified this package. |

Both authors reviewed the code, results and review text, and take responsibility for the judgements in the review.

The model and equations are attributed to Dorrell et al. (2023). The JAX implementation used inspection of the authors' public repository at commit `940659e` to resolve numerical conventions. The NumPy implementation was developed from the paper text. The course's `ccn` report style is used as supplied. NumPy, SciPy, Matplotlib and JAX are installed through Python packages under their respective licences.

Positions and responses are simulated. No private datasets, human participants or animal experiments were used. The implementations have no project redistribution licence; none is implied by this coursework submission.
