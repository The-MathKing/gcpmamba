# Call prep: meeting with Anthony

**Send before the call:** `meeting/project_summary.pdf`, `manuscript/main.pdf`,
`manuscript/supplementary.pdf`, and the reply in `meeting/reply_email.md`.
**Keep open during the call:** `manuscript/BRIEFING.md`, which has the detailed results and likely reviewer questions.

## 1. The 60-second pitch
> Perturb-seq measures how every gene responds when you switch genes on or off, but you can only
> measure a tiny fraction of gene *pairs*. So people build models to predict the rest. Recent papers
> showed that fancy deep models usually don't beat just adding the two single effects together.
> We tried a new idea: a Mamba state-space model that reads all 5,000 genes as a sequence, steered by
> a gene co-expression graph. We tested it very carefully: 5 standard splits, 8 ablations,
> 3 random seeds, proper statistics. It doesn't beat the additive baseline, and the graph doesn't
> help. Along the way we found three evaluation pitfalls that apply to the whole field. One was a
> bug in our own first version, which is why the paper is honest about it.

## 2. Numbers to know by heart (Norman data, error on the top-20 changed genes; lower is better)
| | Error |
|---|---|
| Additive baseline (sum of single effects) | **0.224**, the best |
| GCP-Mamba (ours) | 0.290 |
| Same model without the graph | 0.278, so the graph doesn't help |
| Predict the training mean | 0.428 |
| GEARS, unseen single genes (split 1 only) | 0.214, better than ours (0.262) |

- **Our first version's bug:** ~80% of training perturbations became all-zero inputs.
- **Seed effect:** re-initialising the model moves the error by ~0.046, more than any graph effect.

## 3. Questions to ask (in priority order)
1. **Venue and framing.** Is a careful negative result plus evaluation pitfalls enough for
   *Bioinformatics*? Would *Bioinformatics Advances*, *PLOS Comp Bio*, or an ML-for-biology workshop
   (e.g. MLCB, a NeurIPS workshop) be a better fit? Should we post a bioRxiv preprint first?
2. **Lead with the model or with the pitfalls?** The metric finding (GI Pearson ranks models in
   reverse) may be the most broadly useful result. Should it be the headline?
3. **GEARS comparison.** Is one split with 3 epochs acceptable if clearly labelled? Where could high
   school students get GPU time (lab access, Google Colab Pro, cloud research credits)?
4. **Knowledge graph.** GEARS's Gene Ontology graph helped on unseen genes. Should we swap in GO or
   STRING and rerun, or would that turn this into a different paper?
5. **Statistics.** Is a pooled Wilcoxon over overlapping splits OK, or should we use a mixed-effects
   model or per-split tests?
6. **Another dataset.** The Replogle screens couldn't be used because most of their targets aren't
   measured genes. Is there a standard way around that, or another combinatorial dataset?
7. **Practical.** Would a researcher need to vouch for or co-author the work for it to be taken
   seriously? How should high school authors list affiliations?

## 4. Questions Anthony may ask, and short answers
- *"Why Mamba at all?"* Linear scaling lets it read all 5,000 genes; attention runs out of memory at
  5,000. But we also report honestly that a sparse graph network is 6–25× faster and uses 25–50× less memory.
- *"What's the step-size conditioning?"* Genes close to the perturbed gene in the co-expression
  graph take bigger update steps in the scan. Ablations show it has no measurable effect.
- *"How do you know the model isn't just broken?"* It clearly beats the no-change, mean and linear
  baselines, it is verified against a naive implementation (unit test), and the additive-prior
  variant exactly recovers the additive baseline.
- *"What was the bug?"* Perturbations were encoded only over the 500 most variable genes, and just
  14 of the 105 targets were in that set. We audit it in Supplementary Table S1.
- *"What would make it a positive result?"* Probably a better graph (GO or regulatory), since
  GEARS's GO graph helped on unseen genes. That's the natural next experiment.

## 5. Who covers what on the call
- **Riley:** biology (Perturb-seq, K562, genetic interactions, why additivity is a strong baseline).
- **Aryan:** model, experiments, statistics, code.
- Take notes on: venue advice, compute options, and any analysis he says is essential before submitting.

## 6. After the call
Send a thank-you within a day that lists the 2–3 concrete things he suggested, and ask whether he'd be
willing to look at a revised draft.
