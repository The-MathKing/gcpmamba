# GCP-Mamba: graph-conditioned state-space models for perturbation response prediction

Code and results for *"Graph-conditioned state-space models for genetic perturbation response
prediction: a controlled evaluation against simple baselines"* (Padarthi & Huynh).

GCP-Mamba is a bidirectional selective state-space model (Mamba-style) over all measured genes, in
which the diffusion of a perturbation over a control-cell co-expression graph modulates the
discretization step size of every gene token. It is benchmarked against simple baselines, GEARS and
eight ablations on the standard GEARS splits of the Norman et al. (2019) and Adamson et al. (2016)
Perturb-seq screens.

## Layout

| Path | Contents |
|---|---|
| `gcpm/prepare.py` | pseudobulk responses, control-cell co-expression graph, gene embeddings, Fiedler order |
| `gcpm/make_splits.py` | official GEARS simulation splits (seeds 1-5) |
| `gcpm/model.py` | GCP-Mamba and the chunked parallel selective scan |
| `gcpm/benchmark.py` | baselines, deep-model variants, training, metrics (resumable) |
| `gcpm/run_gears.py` | GEARS retrained on the same splits |
| `gcpm/scaling.py` | measured time and memory versus number of genes |
| `gcpm/report.py` | every table and figure in the manuscript, from `results/` |
| `gcpm/audit_hvg_encoding.py` | audit of the input-encoding flaw in the earlier version (`legacy/`) |
| `results/` | per-perturbation metrics, training curves, summaries |
| `manuscript/` | LaTeX source of the paper and supplement |
| `legacy/` | code and manuscript of the earlier version (kept for transparency; see Section 3.1 of the paper) |

## Reproducing the results

```bash
pip install -r requirements.txt
mkdir -p data && cd data
curl -L -o norman.zip  https://dataverse.harvard.edu/api/access/datafile/6154020 && unzip norman.zip
curl -L -o adamson.zip https://dataverse.harvard.edu/api/access/datafile/6154417 && unzip adamson.zip
cd ..
python -m gcpm.prepare norman && python -m gcpm.prepare adamson
python -m gcpm.make_splits norman && python -m gcpm.make_splits adamson   # splits are also in splits/
./run_queue.sh            # all experiments; resumable, ~1 day on a 4-core CPU
python -m gcpm.report     # tables and figures -> manuscript/
PYTHONPATH=. python tests/test_scan.py
```

The expected SHA-256 of `data/norman/perturb_processed.h5ad` is
`23ffb0fac6a847ff927cf7509d80d85052bfefbfb97610786a2dafaaefa0b6a0`.

## License
MIT — see [LICENSE](LICENSE).
