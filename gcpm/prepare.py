"""Build the cached pseudobulk dataset used by every model in the benchmark.

Everything derived from expression (co-expression graph, gene embeddings, gene
ordering) is computed from unperturbed control cells only, so it is identical
across splits and never sees a held-out perturbation.
"""
import json
import numpy as np
import anndata as ad
import scipy.sparse as sp
from sklearn.decomposition import PCA

H5AD = 'data/norman/perturb_processed.h5ad'
OUT = 'data/norman_pseudobulk.npz'
KNN = 20          # neighbours kept per gene in the co-expression graph
EMB_DIM = 64      # dimension of control-cell gene embeddings


def main():
    a = ad.read_h5ad(H5AD)
    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
    genes = a.var.gene_name.to_numpy()
    gene_ids = a.var.index.to_numpy()
    cond = a.obs.condition.to_numpy()

    ctrl = cond == 'ctrl'
    Xc = X[ctrl].toarray().astype(np.float64)
    ctrl_mean = Xc.mean(0)

    conds = sorted(c for c in np.unique(cond) if c != 'ctrl')
    delta = np.zeros((len(conds), X.shape[1]), np.float32)
    ncells = np.zeros(len(conds), int)
    for k, c in enumerate(conds):
        m = cond == c
        delta[k] = np.asarray(X[m].mean(0)).ravel() - ctrl_mean
        ncells[k] = m.sum()

    # top-20 DE genes (GEARS convention: non-dropout DE vs control)
    cname = dict(zip(a.obs.condition, a.obs.condition_name))
    gid2idx = {g: i for i, g in enumerate(gene_ids)}
    de20 = np.array([[gid2idx[g] for g in a.uns['top_non_dropout_de_20'][cname[c]]]
                     for c in conds])

    # control-cell co-expression graph: symmetric |Pearson| kNN
    Z = (Xc - Xc.mean(0)) / (Xc.std(0) + 1e-8)
    C = np.abs(Z.T @ Z / Z.shape[0])
    np.fill_diagonal(C, 0)
    nn = np.argpartition(-C, KNN, axis=1)[:, :KNN]
    rows = np.repeat(np.arange(C.shape[0]), KNN)
    W = sp.csr_matrix((C[rows, nn.ravel()], (rows, nn.ravel())), shape=C.shape)
    W = W.maximum(W.T).tocoo()

    # spectral (Fiedler) ordering of genes along the co-expression graph
    Wd = sp.csr_matrix(W)
    deg = np.asarray(Wd.sum(1)).ravel() + 1e-8
    Dm = sp.diags(1 / np.sqrt(deg))
    Lsym = sp.eye(Wd.shape[0]) - Dm @ Wd @ Dm
    from scipy.sparse.linalg import eigsh
    vals, vecs = eigsh(Lsym, k=3, sigma=-1e-3, which='LM')
    fiedler = Dm @ vecs[:, np.argsort(vals)[1]]
    order = np.argsort(fiedler)

    # gene embeddings from control cells (PCA loadings), used for zero-shot targets
    emb = PCA(EMB_DIM, random_state=0).fit(Z).components_.T.astype(np.float32)

    np.savez_compressed(OUT, delta=delta, conds=np.array(conds), ncells=ncells,
                        genes=genes, ctrl_mean=ctrl_mean.astype(np.float32),
                        de20=de20, g_row=W.row, g_col=W.col, g_val=W.data.astype(np.float32),
                        order=order, emb=emb)
    print(len(conds), 'conditions,', X.shape[1], 'genes,', W.nnz, 'graph edges')


if __name__ == '__main__':
    main()
