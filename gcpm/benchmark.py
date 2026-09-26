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
PRED_DIR = 'results/predictions'

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
    'GCP-Mamba (Delta only, w_c~N(0,1))': dict(graph='true', order='fiedler', graph_in_delta=True,
                                               graph_in_input=False, block='ssm', wc_init_std=1.0),
    # residual variants: predict delta - additive prior (sum of training singles), i.e. the GI term
    'GCP-Mamba (additive prior)': dict(graph='true', order='fiedler', graph_in_delta=True, graph_in_input=True,
                                       block='ssm', prior=True),
    'Mamba (additive prior)':     dict(graph='none', order='fiedler', graph_in_delta=False, graph_in_input=False,
                                       block='ssm', prior=True),
}
TRAIN = dict(d_model=32, d_state=4, n_layers=2, lr=1e-3, weight_decay=1e-2, batch_size=8,
             max_epochs=100, patience=15, min_epochs=40)


# ───────────────────────────── baselines ─────────────────────────────

LAMBDAS = [10.0 ** k for k in range(-3, 5)]


def baseline_predictions(d, train, val, test):
    """Non-deep baselines; fitted on training conditions, ridge penalty chosen on validation."""
    Y = np.stack([d.delta[d.cidx[c]] for c in train])                     # (n, G)
    singles = observed_singles(d, train + val)
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

def observed_singles(d, observed):
    """{gene: response} for every gene perturbed alone in the observed (training + validation)
    conditions, orientations averaged. GEARS calls a gene 'seen' if it is in either set."""
    allowed = set(observed)
    genes = {targets(c)[0] for c in observed if len(targets(c)) == 1}
    return {g: d.single_response(g, allowed) for g in genes}


def additive_prior(d, observed, conds):
    """Sum of the targets' observed single responses (training + validation); the mean single
    response for a target without one. A single perturbation always gets the mean single response (never its own
    measured response), so the prior is computed identically for training and test conditions."""
    singles = observed_singles(d, observed)
    mean_single = np.stack(list(singles.values())).mean(0)
    out = []
    for c in conds:
        t = targets(c)
        out.append(mean_single if len(t) == 1 else sum(singles.get(g, mean_single) for g in t))
    return np.stack(out).astype(np.float32)


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
        R = additive_prior(d, train + val, conds) if spec.get('prior') else np.zeros_like(Y)
        return [torch.tensor(a[:, order]) for a in (P, C, Y - R, R)]   # target is the residual

    Ptr, Ctr, Ytr, _ = tensors(train)
    Pva, Cva, Yva, _ = tensors(val)
    Pte, Cte, _, Rte = tensors(test)
    model = GCPMamba(G, torch.tensor(d.emb[order]), torch.tensor(d.ctrl_mean[order]),
                     d_model=TRAIN['d_model'], d_state=TRAIN['d_state'], n_layers=TRAIN['n_layers'],
                     graph_in_delta=spec['graph_in_delta'], graph_in_input=spec['graph_in_input'],
                     block=spec['block'], init_mean=Ytr.mean(0),
                     wc_init_std=spec.get('wc_init_std', 0.0))
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
            # validation loss often plateaus for several epochs before the model starts using the
            # perturbation, so early stopping is only allowed after min_epochs
            if bad >= TRAIN['patience'] and ep + 1 >= TRAIN['min_epochs']:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = torch.cat([model(Pte[i:i + 16], Cte[i:i + 16]) for i in range(0, len(Pte), 16)]) + Rte
    out = out.numpy()
    inv = np.argsort(order)
    # diagnostics: norm of the Delta-conditioning weights w_c in each SSM
    wc = {n: p.norm().item() for n, p in model.named_parameters() if n.endswith('w_c')}
    return {c: out[k][inv] for k, c in enumerate(test)}, hist, wc


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
    singles = [d.single_response(g) for g in t]
    if len(t) == 2 and all(x is not None for x in singles):
        # GI residual eps = d_AB - d_A - d_B with measured singles (used only for scoring). The MSE of
        # the predicted residual equals mse_de20, so we store the residual variance for R^2_GI instead.
        add = singles[0] + singles[1]
        m['gi_pearson_de20'] = pearson(pred[de] - add[de], true[de] - add[de])
        m['gi_var_de20'] = float(np.mean((true - add)[de] ** 2))
    return m


def save_predictions(dataset, seed, model, pred, rep=0):
    """Store predicted responses (float32) so every metric can be recomputed without retraining."""
    os.makedirs(PRED_DIR, exist_ok=True)
    conds = sorted(pred)
    safe = ''.join(ch if ch.isalnum() else '_' for ch in model)
    np.savez_compressed(f'{PRED_DIR}/{dataset}_split{seed}_{safe}.npz' if rep == 0 else f'{PRED_DIR}/{dataset}_split{seed}_rep{rep}_{safe}.npz', model=model, conditions=np.array(conds),
                        pred=np.stack([pred[c] for c in conds]).astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    ap.add_argument('--models', nargs='+', default=['all'])
    ap.add_argument('--reps', type=int, nargs='+', default=[0],
                    help='initialisation replicates; replicate r uses seed split_seed + 1000 r')
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

    done = set()
    if os.path.exists(args.out):   # resume: skip (split, model) pairs already written
        prev = pd.read_csv(args.out, usecols=['seed', 'rep', 'model'])
        done = set(zip(prev.seed, prev.rep, prev.model))

    def write(rows, path):
        pd.DataFrame(rows).to_csv(path, mode='a', header=not os.path.exists(path), index=False)

    for seed in args.seeds:
        s2c, sub = load_split(seed, args.dataset)
        train = [c for c in s2c['train'] if c in d.cidx]
        val, test = s2c['val'], s2c['test']
        if do_base:
            preds = baseline_predictions(d, train, val, test)
            for name, p in preds.items():
                save_predictions(args.dataset, seed, name, p)
            write([dict(seed=seed, rep=0, model=name, condition=c, subgroup=sub[c], **condition_metrics(d, c, p[c]))
                   for name, p in preds.items() if (seed, 0, name) not in done for c in test], args.out)
        for rep in args.reps:
            for name in names:
                if (seed, rep, name) in done:
                    continue
                t0 = time.time()
                log(f'[split {seed} rep {rep}] {name}')
                pred, hist, wc = train_deep(d, train, val, test, VARIANTS[name], seed + 1000 * rep, log)
                save_predictions(args.dataset, seed, name, pred, rep)
                write([dict(seed=seed, rep=rep, model=name, condition=c, subgroup=sub[c],
                            **condition_metrics(d, c, pred[c])) for c in test], args.out)
                write([dict(seed=seed, rep=rep, model=name, epoch=e, train=a, val=b) for e, a, b in hist],
                      args.out.replace('.csv', '_curves.csv'))
                if wc:
                    write([dict(seed=seed, rep=rep, model=name, param=k, norm=v) for k, v in wc.items()],
                          args.out.replace('.csv', '_wc_norms.csv'))
                log(f'    done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
