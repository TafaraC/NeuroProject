# Evidence behind the paper review

This is an editing aid, not additional review text or a substitute for the
students' own assessment. No prior peer reviews were used.

| Review point | Primary evidence | Interpretation boundary |
| --- | --- | --- |
| Linear, location-independent action updates | Dorrell et al., equation 3 and section 2 | A modelling choice; not a demonstrated biological update rule |
| Fourier form and finite frequency budget | Equation 6; Appendix A.2-A.4 | The plane is treated through a large-torus approximation |
| Frequency lattices and modules | Sections 3.1-3.3; Figures 3-4; Appendix E | Several arguments are heuristic, with numerical support |
| Limits of the optimality claim | Appendix G | Non-convex frequency choice is acknowledged; do not describe the full result as a global-optimality proof |
| Module-count sensitivity | Appendix B.4.2, Figure 9 | The paper acknowledges reduced hexagonality in some multi-module cases |
| Field width, module orientation, room geometry | Section 4, Figure 5, Appendices J-L | Biological comparisons are described as preliminary/rudimentary by the authors |
| Norm and optimisation ambiguities | Equations 22-28; table B.3; public code commit 940659ed2773f4cbb3bcf298bdb70497b30d250c | Report actual differences; do not infer which code generated a published panel |
| Earlier mechanistic account | Burak & Fiete (2009), PLoS Computational Biology 5(2):e1000291, https://doi.org/10.1371/journal.pcbi.1000291 | Explains path integration in continuous attractors; a different explanatory level |
| Earlier task-trained account | Banino et al. (2018), Nature 557:429-433, https://doi.org/10.1038/s41586-018-0102-6 | Grid-like units and navigation preceded this paper; the unification is the relevant novelty |
| Population geometry in recordings | Gardner et al. (2022), Nature 602:123-128, https://doi.org/10.1038/s41586-021-04268-7 | Toroidal structure supports relevance of population geometry, not uniquely Dorrell's objective |
| Original reproduction outcome | results/reference_runs/README.md and both saved evaluation summaries | One seed per spatial-scale condition; explicit deviations; not evidence of irreproducibility |

Paper bibliographic record: https://arxiv.org/abs/2209.15563 (v3, 28 February
2023; published at ICLR 2023). Author order is William Dorrell, Peter E. Latham,
Timothy E. J. Behrens, James C. R. Whittington.

The proposed experiment separating physical geometry and occupancy is a
forward-looking recommendation in the review, not an experiment conducted here.
The review uses four scholarly references and the authors' public code link.

Checklist source: https://neurips.cc/public/guides/PaperChecklist, consulted
25 October 2026. Its 16 topics are paraphrased and answered for our reproduction,
not on behalf of the reviewed authors. Personal ethics declarations and the
official Wits AI form require the students' confirmation.
