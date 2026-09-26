"""Benchmark on the Norman et al. (2019) GEARS simulation splits.

Usage:
    python -m gcpm.benchmark --seeds 1 2 3 4 5 --models all
Writes one row per (split seed, model, test condition) to results/<dataset>_per_condition.csv.
"""
import argparse
import os
import time
import numpy as np
import pandas as pd
import torch

from gcpm.data import NormanData, load_split, targets
from gcpm.model import GCPMamba

OUT = 'results/{}_per_condition.csv'

# Deep-model variants. Every variant receives the same perturbation information
# (target-token flag + control-cell target embedding); they differ only in how
# the co-expression graph is used and whether genes interact through the scan.
VARIANTS = {
    'GCP-Mamba':              dict(graph='true', order='fiedler', graph_in_delta=True, graph_in_input=True, block='ssm'),
    'GCP-Mamba (Delta only)': dict(graph='true', order='fiedler', graph_in_delta=True, graph_in_input=False, block='ssm'),
    'GCP-Mamba (perm. graph)': dict(graph='perm', order='fiedler', graph_in_delta=True, graph_in_input=True, block='ssm'),
    'GCP-Mamba (random order)': dict(graph='true', order='random', graph_in_delta=True, graph_in_input=True, block='ssm'),
    'Mamba (no graph)':       dict(graph='none', order='fiedler', graph_in_delta=False, graph_in_input=False, block='ssm'),
    'Graph-MLP (no scan)':    dict(graph='true', order='fiedler', graph_in_delta=True, graph_in_input=True, block='mlp'),
}
TRAIN = dict(d_model=32, d_state=4, n_layers=2, lr=1e-3, weight_decay=1e-2, batch_size=8,
             max_epochs=100, patience=15)


# ───────────────────────────── baselines ─────────────────────────────

LAMBDAS = [10.0 ** k for k in range(-3, 5)]


def baseline_predictions(d, train, val, test):
    """Non-deep baselines; fitted on training conditions, ridge penalty chosen on validation."""
    Y = np.stack([d.delta[d.cidx[c]] for c in train])                     # (n, G)
    singles = {targets(c)[0]: d.delta[d.cidx[c]] for c in train if len(targets(c)) == 1}
    mean_all = Y.mean(0)
    mean_single = np.stack(list(singles.values())).mean(0)
    preds = {'No change': {}, 'Train mean': {}, 'Additive': {}}
    for c in test:
        preds['No change'][c] = np.zeros_like(mean_all)
        preds['Train mean'][c] = mean_all
        preds['Additive'][c] = sum(singles.get(g, mean_single) for g in targets(c))

    # Ahlmann-Eltze et al. (2025) linear model: Y ~ G W P^T + b, with gene
    # embeddings G from a PCA of the training responses and each perturbation
    # represented by the sum of its targets' embeddings. Two choices of the
    # target embedding: the training-PCA rows (as in the original) and the
    # control-cell embedding that GCP-Mamba itself receives.
    b = Y.mean(0)
    Yc = (Y - b).T                                                        # (G, n)
    U, S, _ = np.linalg.svd(Yc, full_matrices=False)
    G = U[:, :10]
    Yv = np.stack([d.delta[d.cidx[c]] for c in val])
    for name, E in (('Linear (PCA)', U[:, :10]), ('Linear (ctrl emb.)', d.emb)):
        emb = lambda c: sum(E[d.gidx[g]] for g in targets(c))
        P = np.stack([emb(c) for c in train])
        best = None
        for lam in LAMBDAS:
            W = np.linalg.solve(G.T @ G + lam * np.eye(G.shape[1]), G.T @ Yc @ P) \
                @ np.linalg.inv(P.T @ P + lam * np.eye(P.shape[1]))
            err = np.mean((np.stack([G @ W @ emb(c) + b for c in val]) - Yv) ** 2)
            if best is None or err < best[0]:
                best = (err, W)
        preds[name] = {c: G @ best[1] @ emb(c) + b for c in test}
    return preds


# ───────────────────────────── deep models ─────────────────────────────

def train_deep(d, train, val, test, spec, seed, log):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    G = len(d.genes)
    order = d.order if spec['order'] == 'fiedler' else rng.permutation(G)
    W = d.W
    if spec['graph'] == 'perm':        # shuffle gene labels of the graph
        pi = rng.permutation(G)
        W = W[pi][:, pi]

    def tensors(conds):
        P = np.stack([d.indicator(c) for c in conds])
        C = d.graph_signal(P, W) if spec['graph'] != 'none' else np.zeros_like(P)
        Y = np.stack([d.delta[d.cidx[c]] for c in conds])
        return [torch.tensor(a[:, order]) for a in (P, C, Y)]

    Ptr, Ctr, Ytr = tensors(train)
    Pva, Cva, Yva = tensors(val)
    Pte, Cte, _ = tensors(test)
    model = GCPMamba(G, torch.tensor(d.emb[order]), torch.tensor(d.ctrl_mean[order]),
                     d_model=TRAIN['d_model'], d_state=TRAIN['d_state'], n_layers=TRAIN['n_layers'],
                     graph_in_delta=spec['graph_in_delta'], graph_in_input=spec['graph_in_input'],
                     block=spec['block'], init_mean=Ytr.mean(0))
    opt = torch.optim.AdamW(model.parameters(), lr=TRAIN['lr'], weight_decay=TRAIN['weight_decay'])
    best, best_state, bad, hist = np.inf, None, 0, []
    for ep in range(TRAIN['max_epochs']):
        model.train()
        perm = torch.randperm(len(Ptr))
        tl = 0.0
        for i in range(0, len(perm), TRAIN['batch_size']):
            idx = perm[i:i + TRAIN['batch_size']]
            loss = torch.mean((model(Ptr[idx], Ctr[idx]) - Ytr[idx]) ** 2)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            vl = torch.mean((model(Pva, Cva) - Yva) ** 2).item()
        hist.append((ep, tl / len(Ptr), vl))
        log(f'    ep {ep:2d} train {tl / len(Ptr):.5f} val {vl:.5f}')
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= TRAIN['patience']:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = torch.cat([model(Pte[i:i + 16], Cte[i:i + 16]) for i in range(0, len(Pte), 16)]).numpy()
    inv = np.argsort(order)
    return {c: out[k][inv] for k, c in enumerate(test)}, hist


# ───────────────────────────── metrics ─────────────────────────────

def pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 1e-12 else np.nan


def condition_metrics(d, c, pred):
    true = d.delta[d.cidx[c]]
    de = d.de20[d.cidx[c]]
    m = dict(mse_de20=float(np.mean((pred[de] - true[de]) ** 2)),
             pearson_delta=pearson(pred, true),
             pearson_delta_de20=pearson(pred[de], true[de]),
             direction_de20=float(np.mean(np.sign(pred[de]) == np.sign(true[de]))))
    t = targets(c)
    if len(t) == 2 and all(f'{g}+ctrl' in d.cidx or f'ctrl+{g}' in d.cidx for g in t):
        add = sum(d.delta[d.cidx[f'{g}+ctrl' if f'{g}+ctrl' in d.cidx else f'ctrl+{g}']] for g in t)
        m['gi_pearson_de20'] = pearson(pred[de] - add[de], true[de] - add[de])
        m['gi_mse_de20'] = float(np.mean(((pred - add)[de] - (true - add)[de]) ** 2))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    ap.add_argument('--models', nargs='+', default=['all'])
    ap.add_argument('--dataset', default='norman')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    args.out = args.out or OUT.format(args.dataset)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.set_num_threads(os.cpu_count())
    d = NormanData(args.dataset)
    names = list(VARIANTS) if args.models == ['all'] else [m for m in args.models if m in VARIANTS]
    do_base = args.models == ['all'] or 'baselines' in args.models
    logf = open(args.out.replace('.csv', '.log'), 'a')

    def log(s):
        print(s, flush=True)
        logf.write(s + '\n')
        logf.flush()

    for seed in args.seeds:
        s2c, sub = load_split(seed, args.dataset)
        train = [c for c in s2c['train'] if c in d.cidx]
        val, test = s2c['val'], s2c['test']
        preds = baseline_predictions(d, train, val, test) if do_base else {}
        curves = []
        for name in names:
            t0 = time.time()
            log(f'[split {seed}] {name}')
            preds[name], hist = train_deep(d, train, val, test, VARIANTS[name], seed, log)
            curves += [dict(seed=seed, model=name, epoch=e, train=a, val=b) for e, a, b in hist]
            log(f'    done in {time.time() - t0:.0f}s')
        rows = [dict(seed=seed, model=name, condition=c, subgroup=sub[c], **condition_metrics(d, c, p[c]))
                for name, p in preds.items() for c in test]
        df = pd.DataFrame(rows)
        df.to_csv(args.out, mode='a', header=not os.path.exists(args.out), index=False)
        if curves:
            cf = args.out.replace('.csv', '_curves.csv')
            pd.DataFrame(curves).to_csv(cf, mode='a', header=not os.path.exists(cf), index=False)
        log(df.groupby('model')[['mse_de20', 'pearson_delta_de20', 'gi_pearson_de20']].mean().to_string())


if __name__ == '__main__':
    main()
