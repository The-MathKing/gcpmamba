# Briefing: what changed in this revision and why

For discussing the manuscript with a reviewer or mentor. Files: `manuscript/main.pdf`,
`manuscript/supplementary.pdf`; every number is regenerated from `results/` by `python -m gcpm.report`.

## 1. The earlier version's result was an artefact
The earlier model encoded a perturbation as a 0/1 vector over the 500 most variable genes. Only 12–17
of the 105 targeted genes were among them, and singles written `ctrl+GENE` were parsed as "control".
So ~80% of training perturbations reached the model as an **all-zero input** (161–183 training
conditions shared only 14–19 distinct inputs). The model could at best learn the average response;
its "the graph doesn't help" conclusion said nothing about the architecture.
Script: `gcpm/audit_hvg_encoding.py` (Supplementary Table S1).

## 2. What was rebuilt
- Every target is represented (its own gene token, plus an embedding from control cells so that
  never-seen targets can be predicted).
- The graph signal is gene- and perturbation-specific. In the old model it collapsed to one global
  vector, so the "permuted graph" control could not differ from the real graph by construction.
- Bidirectional selective scan over all 5,045 genes, verified against a naive recurrence (unit test).
- Standard GEARS splits (5 seeds) of the Norman data, plus the Adamson screen.
- Baselines: no change, train mean, additive, linear (Ahlmann-Eltze 2025), GEARS (one split, 3 epochs:
  CPU-limited).
- 8 ablations; 3 initialisations per split for the main variants; Holm-corrected Wilcoxon tests.

## 3. Results (honest summary)
- **Additive baseline wins on doubles** (MSE 0.224 vs 0.290 for GCP-Mamba). Predicting only the
  deviation from additive matches it (0.225) but explains no interaction variance (R²_GI ≈ 0).
- **Unseen singles:** everything reasonable is within ~4% of the mean response; GCP-Mamba is not
  significantly better than the mean, the linear model or its graph-free/permuted ablations.
- **The graph and the step-size conditioning have no measurable effect.**
- **GEARS (split 1 only, 3 epochs):** best on unseen targets (unseen singles 0.214 vs 0.262 for
  GCP-Mamba) and better than GCP-Mamba overall (P = 0.003), but still worse than additive overall
  (0.230 vs 0.213). Suggests Gene Ontology graphs carry information co-expression does not.
- **Adamson:** all methods within noise of the mean response (low-signal dataset).
- **Cost:** the scan scales linearly (attention runs out of memory at 5,000 genes), but a sparse
  graph network is 6–25× faster and uses 25–50× less memory.

## 4. Two general lessons (the paper's main contribution)
1. Audit the perturbation encoding — count how many targets reach the input.
2. A single-seed comparison showed a "significant" graph effect that vanished with 3 seeds: the
   initialisation alone moves the error by ~0.046, more than any graph effect (Supp. Table S4).
3. The genetic-interaction *Pearson* metric used in earlier work ranks models in reverse: trivial
   predictors score like trained models, and the models closest to the truth score lowest. Use
   variance explained relative to the additive prediction instead (R²_GI).

## 5. Questions a reviewer is likely to ask
- *Why only K562 screens?* Replogle screens were excluded because most targets are not measured
  genes in the GEARS release, which the token encoding needs (stated in Limitations).
- *Why is GEARS only one split, 3 epochs?* CPU-only compute: ~40 min per epoch, and its built-in
  evaluations had to be skipped to fit in 15 GB RAM (predictions are scored identically to all other
  models). Stated as a limitation; a GPU rerun on all 5 splits would strengthen this comparison.
- *Isn't it bad that GEARS beats your model?* It supports the paper's explanation: the co-expression
  prior is the weak part; a knowledge graph helps unseen targets. It is reported, not hidden.
- *Would tuning help?* Hyperparameters were chosen on split-1 validation only; no variant approached
  the additive baseline, so tuning is unlikely to reverse the conclusion.
- *Is it publishable if negative?* As a careful evaluation with reusable pitfalls, yes in principle;
  Bioinformatics publishes such studies, but acceptance is never guaranteed.
