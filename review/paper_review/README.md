# Review draft

Main source: `main.tex`. Uses the unmodified `ccn.sty` supplied in the course
template; the template's blank placeholder pages and unused packages are omitted.
The review has six rubric sections. References, supporting figure/table,
contribution statement, AI disclosure, and checklist follow the main text.

Tafara Chibanda and Nathan Le Roux confirmed the review's judgements and their
contributions on 26 October 2026. The contribution statement elsewhere in
`review/` has been updated to match their independent implementations.

Tafara will append the completed official Wits Science Faculty AI Declaration.
Use `ai_declaration.tex`, included at the end of `main.tex`, then recompile and
replace Review.pdf in the submission package. The assistance disclosure is not
a substitute for the official declaration.

Measured findings include the preserved seed-0 spatial-scale runs under
`results/reference_runs`, nine completed matched-seed runs under
`results/diagnostic_study`, the shared-input audit in `results/partner_comparison`,
and the constructed (not trained) code in `results/constraint_control`.
The final joint review also includes Nathan's runs at commit df44d69 and their
common evaluation in `results/partner_results_20261026`.
The diagnostic experiments are separate from the original Figure 8 comparison.

Build with a TeX installation supporting the supplied CCN style, e.g.
`tectonic main.tex`, or use Codex's built-in LaTeX project compiler.
