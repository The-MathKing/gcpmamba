**To:** Anthony
**Cc:** Aryan Padarthi
**Subject:** Re: Intro — GCP-Mamba perturbation-prediction project

Hi Anthony,

Thank you so much for making time for us, and thanks to Kevin for the introduction!

Monday 9/28 after school works well for us. Would [TIME, e.g. 5:00 pm CT] work for you? We're happy to
adjust to whatever suits you. I've cc'd my co-author, Aryan Padarthi, who led the modelling side; I
worked on the biology and writing.

Some background ahead of the call:

- **One-page summary** (attached): the problem, what we built, and our main results.
- **Full draft** (attached, 7 pages + supplement), which we're preparing for *Bioinformatics*.
- **Code and results:** https://github.com/The-MathKing/gcpmamba

In short, we built a graph-conditioned state-space model (Mamba-style) to predict transcriptome-wide
responses to CRISPR perturbations, and benchmarked it carefully against simple baselines. It turned out
*not* to beat the "sum of single effects" baseline. The paper is now framed as a careful negative result,
with three evaluation pitfalls we ran into, including a bug in our own first version.

The specific questions we're working through:

1. Is this negative/evaluation framing strong enough for *Bioinformatics*, or would you suggest a
   different venue or framing?
2. Our GEARS comparison is limited to one split and shortened training because we only have CPU
   access. How much does that weaken the paper, and is there a realistic way to get GPU time?
3. We propose a new way to score genetic-interaction predictions (variance explained relative to the
   additive prediction) instead of the correlation metric used before. Does that seem sound, or is
   there an established alternative we missed?
4. Are our statistics appropriate? We use Wilcoxon tests pooled over overlapping data splits, with
   Holm correction.

No need to read everything; the one-page summary and the questions above are the most useful part.
Looking forward to talking!

Best,
Riley
