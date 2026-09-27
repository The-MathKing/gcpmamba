Dear Editor,

We submit our manuscript "Graph-conditioned state-space models for genetic perturbation response
prediction: a controlled evaluation against simple baselines" for consideration as an Original Paper
in *Bioinformatics* (Gene expression).

Predicting transcriptional responses to unmeasured genetic perturbations is a central problem in
single-cell perturbation modelling, and recent work (Ahlmann-Eltze et al., *Nature Methods* 2025) has
shown that deep models rarely outperform simple baselines. We asked whether a linear-time state-space
model conditioned on a gene–gene co-expression graph closes this gap. Rather than reporting a single
favourable comparison, we evaluated the model on the five standard GEARS splits of the Norman et al.
screen and on the Adamson et al. screen, against five simple baselines and GEARS, with eight ablations
that isolate each component and three initialisations per split.

The answer is negative, and we believe the way we reached it is useful to the field:

- The additive sum of single-perturbation effects remained the best predictor for double
  perturbations; no model explained genetic-interaction variance beyond additivity.
- The graph and the graph-conditioned discretization gave no reliable benefit; an apparent benefit in
  a single-seed comparison disappeared with replicate initialisations.
- We document two evaluation pitfalls: a perturbation encoding restricted to highly variable genes
  that silently gave most perturbations an all-zero input (in an earlier version of our own work),
  and a genetic-interaction correlation metric that ranks models in the reverse order of the
  variance they explain. We propose a variance-explained alternative.

All code, splits, per-perturbation results and scripts that regenerate every table and figure are
openly available. The manuscript has not been published or submitted elsewhere. Both authors approved
the submission and declare no conflicts of interest.

Sincerely,
Aryan Padarthi and Riley Huynh
Allen High School, Allen, TX, USA
