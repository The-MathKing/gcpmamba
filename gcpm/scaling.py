"""Measured cost of one training step (forward + backward) versus number of gene tokens L.

Each configuration runs in a fresh subprocess and reports wall time and peak
resident memory above the post-import baseline. Compared layers (d_model=32,
batch 8, 2 layers each):
  gcp-mamba : bidirectional graph-conditioned selective scan (this work)
  attention : dense multi-head self-attention over genes (Transformer-style)
  sparse-gnn: 2-layer GCN on a k=20 nearest-neighbour gene graph (GEARS-style)

Usage: python -m gcpm.scaling  -> results/scaling.csv
"""
import json
import os
import resource
import subprocess
import sys
import time

import pandas as pd

LENGTHS = [500, 1000, 2000, 5000, 10000, 20000]
MODELS = ['gcp-mamba', 'attention', 'sparse-gnn']
B, D, K = 8, 32, 20


def run_one(model, L):
    import torch
    import torch.nn as nn
    from gcpm.model import BiSSMBlock
    torch.manual_seed(0)
    torch.set_num_threads(os.cpu_count())
    x = torch.randn(B, L, D, requires_grad=True)
    c = torch.rand(B, L)
    if model == 'gcp-mamba':
        layers = nn.ModuleList([BiSSMBlock(D, 8, True) for _ in range(2)])
        f = lambda h: [h := layer(h, c) for layer in layers][-1]
    elif model == 'attention':
        layers = nn.ModuleList([nn.TransformerEncoderLayer(D, 4, 2 * D, batch_first=True) for _ in range(2)])
        f = lambda h: [h := layer(h) for layer in layers][-1]
    else:
        idx = torch.randint(0, L, (L * K,))
        rows = torch.arange(L).repeat_interleave(K)
        A = torch.sparse_coo_tensor(torch.stack([rows, idx]), torch.full((L * K,), 1.0 / K), (L, L)).coalesce()
        lins = nn.ModuleList([nn.Linear(D, D) for _ in range(2)])

        def f(h):
            for lin in lins:
                h = torch.relu(lin(torch.stack([torch.sparse.mm(A, h[b]) for b in range(B)])))
            return h
    base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t = time.time()
    f(x).sum().backward()
    dt = time.time() - t
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return dict(model=model, L=L, seconds=dt, peak_mb=(peak - base) / 1024)


def main():
    if len(sys.argv) == 3:
        print(json.dumps(run_one(sys.argv[1], int(sys.argv[2]))))
        return
    rows = []
    for m in MODELS:
        for L in LENGTHS:
            p = subprocess.run([sys.executable, '-m', 'gcpm.scaling', m, str(L)],
                               capture_output=True, text=True, timeout=3600)
            if p.returncode == 0:
                rows.append(json.loads(p.stdout.strip().splitlines()[-1]))
            else:   # out of memory (the machine has 15 GB)
                rows.append(dict(model=m, L=L, seconds=float('nan'), peak_mb=float('nan'), failed=True))
            print(rows[-1], flush=True)
    os.makedirs('results', exist_ok=True)
    pd.DataFrame(rows).to_csv('results/scaling.csv', index=False)


if __name__ == '__main__':
    main()
