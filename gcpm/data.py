"""Split-aware access to the cached Norman pseudobulk data (see prepare.py)."""
import json
import numpy as np
import scipy.sparse as sp

NPZ = 'data/{}_pseudobulk.npz'
SPLITS = 'splits/gears_simulation_splits{}.json'   # '' for norman, '_adamson' etc.
DIFFUSION_STEPS = 3
DIFFUSION_DECAY = 0.5
SIGNAL_SCALE = 1e3   # c is compressed as log1p(SIGNAL_SCALE * c); raw values span ~1e-5..1e-1


def targets(cond):
    return [g for g in cond.split('+') if g != 'ctrl']


class NormanData:
    def __init__(self, dataset='norman'):
        d = np.load(NPZ.format(dataset), allow_pickle=True)
        self.delta = d['delta']
        self.conds = list(d['conds'])
        self.cidx = {c: i for i, c in enumerate(self.conds)}
        self.genes = d['genes']
        self.gidx = {g: i for i, g in enumerate(self.genes)}
        self.ctrl_mean = d['ctrl_mean']
        self.de20 = d['de20']
        self.order = d['order']
        # PCA loadings have unit norm over ~5k genes (entries ~0.01); standardise columns
        self.emb = (d['emb'] / d['emb'].std(0)).astype(np.float32)
        G = len(self.genes)
        self.W = sp.csr_matrix((d['g_val'], (d['g_row'], d['g_col'])), shape=(G, G))
        self.ncells = d['ncells']

    def single_response(self, gene, allowed=None):
        """Response to perturbing `gene` alone: the mean over its single-perturbation conditions
        ('GENE+ctrl' and 'ctrl+GENE' are distinct guides for the same target). Returns None if no
        such condition is available (optionally restricted to the conditions in `allowed`)."""
        conds = [c for c in (f'{gene}+ctrl', f'ctrl+{gene}') if c in self.cidx and (allowed is None or c in allowed)]
        return self.delta[[self.cidx[c] for c in conds]].mean(0) if conds else None

    def indicator(self, cond):
        p = np.zeros(len(self.genes), np.float32)
        for g in targets(cond):
            p[self.gidx[g]] = 1.0
        return p

    def graph_signal(self, P, W):
        """c(P) = log1p(s * sum_{k=1..K} decay^{k-1} Wn^k P), Wn the symmetrically normalised graph."""
        deg = np.asarray(W.sum(1)).ravel() + 1e-8
        Dm = sp.diags(1 / np.sqrt(deg))
        Wn = Dm @ W @ Dm
        out, cur = np.zeros_like(P), P.T
        for k in range(DIFFUSION_STEPS):
            cur = Wn @ cur
            out += (DIFFUSION_DECAY ** k) * cur.T
        return np.log1p(SIGNAL_SCALE * out).astype(np.float32)


def load_split(seed, dataset='norman'):
    s = json.load(open(SPLITS.format('' if dataset == 'norman' else '_' + dataset)))[str(seed)]
    sub = {}
    for group, conds in s['subgroup']['test_subgroup'].items():
        for c in conds:
            sub[c] = group
    return s['set2conditions'], sub
