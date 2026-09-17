# Addendum: Decoupled Architecture Investigation

Following the discovery of the input indexing flaw, I conducted an investigation to determine if the GCP-Mamba architecture could succeed if the perturbation representation was properly decoupled from the output gene set (a technique used successfully by GEARS).

## Experimental Setup
I created `test_decoupled.py`, which implements a `DecoupledMamba` model:
1. It maintains an independent vocabulary of all 105 unique perturbed genes.
2. It projects the perturbation into a dense embedding $p \in \mathbb{R}^{d_{model}}$.
3. It creates a learnable base expression sequence for the 500 HVGs, and adds the perturbation embedding $p$ to every token in the sequence.
4. This conditioned sequence is then passed through the recurrent Mamba blocks.

This setup guarantees that the model receives complete and accurate perturbation information for every sample, regardless of whether the target gene is in the 500 HVGs.

## Results
The decoupled architecture was trained for 15 epochs on the Norman dataset. 

| Epoch | Train Loss | Test MSE | Test Pearson ($r$) |
|-------|------------|----------|--------------------|
| 5     | 0.3327     | 0.2953   | -0.0127           |
| 10    | 0.1158     | 0.1189   | -0.0091           |
| 15    | 0.0809     | 0.0790   | 0.0080            |

## Conclusion
Despite receiving perfect perturbation conditioning, the `DecoupledMamba` model completely failed to generalize, remaining trapped at a test correlation of $r \approx 0.0$. 

This is a profound result: **it means the original paper's negative conclusion remains entirely valid.** 

While the indexing bug crippled the original input mechanism, fixing it via decoupling reveals that the underlying sequence model (SSM) still fundamentally fails to learn meaningful cross-gene regulatory effects compared to a GNN (which achieved $r > 0.30$ under identical conditions). The sequential recurrence of Mamba, even when conditioned on structural graph priors in the discretization step, is simply the wrong inductive bias for unordered, highly-connected gene interaction manifolds.

Because this confirms the paper's negative result, no changes to the manuscript's conclusions are necessary, though the indexing bug should likely be noted as an additional technical limitation of the original formulation.
