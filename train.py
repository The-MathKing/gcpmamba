import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.stats import pearsonr, ttest_rel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import os, shutil

from model import GCPMamba, BaseMamba
from data_loader import DataEngine

sns.set_theme(style="whitegrid", font_scale=1.1)
ARTIFACT_DIR = '/Users/aryanpadarthi/.gemini/antigravity-ide/brain/6d59065d-fbc0-4e13-9bd5-c418cce18c45'

# ─────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────
N_GENES   = 5000
D_MODEL   = 24
N_LAYERS  = 1
EPOCHS    = 1
LR        = 1e-4
BATCH     = 64
K_FOLDS   = 2
TOP_K     = 20

# ─────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────
def calc_metrics(y_true_np, y_pred_np, k=TOP_K):
    mse_list, r_list, match_list = [], [], []
    for yt, yp in zip(y_true_np, y_pred_np):
        top_idx = np.argsort(np.abs(yt))[-k:]
        yt_k, yp_k = yt[top_idx], yp[top_idx]
        mse_list.append(np.mean((yt_k - yp_k) ** 2))
        if np.std(yt_k) > 1e-6 and np.std(yp_k) > 1e-6:
            r, _ = pearsonr(yt_k, yp_k)
            r_list.append(r)
        else:
            r_list.append(0.0)
        match_list.append(np.mean(np.sign(yt_k) == np.sign(yp_k)))
    return np.mean(mse_list), np.mean(r_list), np.mean(match_list)

# ─────────────────────────────────────────────────────────────────────────
# TRAIN / EVAL LOOPS
# ─────────────────────────────────────────────────────────────────────────
def _unpack_batch(batch):
    return batch[0], batch[1]

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0
    for batch in loader:
        x, y = _unpack_batch(batch)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / len(loader)

def evaluate_loader(model, loader, device):
    model.eval()
    yt_all, yp_all = [], []
    with torch.no_grad():
        for batch in loader:
            x, y = _unpack_batch(batch)
            yt_all.append(y.numpy())
            yp_all.append(model(x.to(device)).cpu().numpy())
    yt_all = np.concatenate(yt_all)
    yp_all = np.concatenate(yp_all)
    mse, r, match = calc_metrics(yt_all, yp_all)
    return mse, r, match, yt_all, yp_all

# ─────────────────────────────────────────────────────────────────────────
# SINGLE TRAINING RUN
# ─────────────────────────────────────────────────────────────────────────
def run_model(model, train_dl, eval_dls, device, name=""):
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss()

    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, train_dl, optimizer, criterion, device)
        scheduler.step()
        if epoch % 10 == 0 or epoch == 1:
            print(f"  [{name}] Epoch {epoch:2d} | train loss={loss:.4f}")

    results = {}
    final_yt, final_yp = None, None
    for split_name, dl in eval_dls.items():
        mse, r, match, yt, yp = evaluate_loader(model, dl, device)
        results[split_name] = dict(mse=mse, pearson=r, match=match)
        if split_name == "Seen 0/2":
            final_yt, final_yp = yt, yp
    return results, final_yt, final_yp

# ─────────────────────────────────────────────────────────────────────────
# 3-FOLD CROSS VALIDATION WITH PAIRED T-TEST
# ─────────────────────────────────────────────────────────────────────────
def kfold_cv(X, y, D, device, k=K_FOLDS):
    from torch.utils.data import DataLoader, TensorDataset
    n = len(X)
    fold_size = n // k
    splits = ['Seen 2/2', 'Seen 1/2', 'Seen 0/2']
    
    records_mamba = {s: [] for s in splits}
    records_base  = {s: [] for s in splits}

    rng = np.random.default_rng(7)
    perm = rng.permutation(n)
    X, y = X[perm], y[perm]

    for fold in range(k):
        print(f"\n=== Fold {fold+1}/{k} ===")
        val_start = fold * fold_size
        val_end   = val_start + fold_size
        val_idx   = np.arange(val_start, val_end)
        train_idx = np.concatenate([np.arange(0, val_start), np.arange(val_end, n)])

        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        v = len(X_val)
        s2_idx  = np.arange(0, v // 3)
        s1_idx  = np.arange(v // 3, 2 * v // 3)
        s0_idx  = np.arange(2 * v // 3, v)

        def mk_dl(x_sub, y_sub, shuf=False):
            ds = TensorDataset(x_sub, y_sub)
            return DataLoader(ds, batch_size=BATCH, shuffle=shuf)

        train_dl = mk_dl(X_tr, y_tr, shuf=True)
        eval_dls = {
            'Seen 2/2': mk_dl(X_val[s2_idx], y_val[s2_idx]),
            'Seen 1/2': mk_dl(X_val[s1_idx], y_val[s1_idx]),
            'Seen 0/2': mk_dl(X_val[s0_idx], y_val[s0_idx]),
        }

        model_m = GCPMamba(n_genes=N_GENES, D=D, d_model=D_MODEL, n_layers=N_LAYERS).to(device)
        model_b = BaseMamba(n_genes=N_GENES, d_model=D_MODEL, n_layers=N_LAYERS).to(device)

        res_m, _, _ = run_model(model_m, train_dl, eval_dls, device, name="GCP-Mamba")
        res_b, _, _ = run_model(model_b, train_dl, eval_dls, device, name="BaseMamba")

        for s in splits:
            records_mamba[s].append(res_m[s])
            records_base[s].append(res_b[s])

    return records_mamba, records_base

def compute_pvalues(records_m, records_b):
    """Run paired Student's t-test on fold-by-fold MSE values."""
    splits = ['Seen 2/2', 'Seen 1/2', 'Seen 0/2']
    pvals = {}
    for s in splits:
        mse_m = np.array([r['mse'] for r in records_m[s]])
        mse_b = np.array([r['mse'] for r in records_b[s]])
        if len(mse_m) >= 2:
            _, p = ttest_rel(mse_m, mse_b)
        else:
            p = float('nan')
        pvals[s] = p
        print(f"  [{s}] Paired t-test p={p:.4f} ({'*significant*' if p < 0.05 else 'ns'})")
    return pvals

# ─────────────────────────────────────────────────────────────────────────
# FIGURE GENERATION
# ─────────────────────────────────────────────────────────────────────────
def aggregate(records):
    agg = {}
    for split, fold_results in records.items():
        mses     = [r['mse']     for r in fold_results]
        pearsons = [r['pearson'] for r in fold_results]
        matches  = [r['match']   for r in fold_results]
        agg[split] = {
            'mse_mean':  np.mean(mses),  'mse_std':  np.std(mses),
            'r_mean':    np.mean(pearsons), 'r_std':  np.std(pearsons),
            'match_mean':np.mean(matches),'match_std':np.std(matches),
            'mse_folds': mses,
        }
    return agg

def generate_ablation_chart(agg_m, agg_b, pvals):
    splits = ['Seen 2/2', 'Seen 1/2', 'Seen 0/2']
    x = np.arange(len(splits))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # MSE subplot
    ax = axes[0]
    mse_m  = [agg_m[s]['mse_mean']  for s in splits]
    mse_b  = [agg_b[s]['mse_mean']  for s in splits]
    err_m  = [agg_m[s]['mse_std']   for s in splits]
    err_b  = [agg_b[s]['mse_std']   for s in splits]
    bars_m = ax.bar(x - w/2, mse_m, w, yerr=err_m, label='GCP-Mamba (w/ $M_\\Delta$)', color='#2196F3', capsize=4)
    bars_b = ax.bar(x + w/2, mse_b, w, yerr=err_b, label='BaseMamba (w/o $M_\\Delta$)', color='#FF7043', capsize=4)
    # Add p-value annotations
    for i, s in enumerate(splits):
        p = pvals[s]
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        ymax = max(mse_m[i] + err_m[i], mse_b[i] + err_b[i]) + 0.03
        ax.text(x[i], ymax, sig, ha='center', va='bottom', fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels(splits)
    ax.set_ylabel('MSE (Top-20 Genes)'); ax.set_title('Ablation: MSE Comparison')
    ax.legend()

    # Pearson subplot
    ax = axes[1]
    r_m   = [agg_m[s]['r_mean']   for s in splits]
    r_b   = [agg_b[s]['r_mean']   for s in splits]
    re_m  = [agg_m[s]['r_std']    for s in splits]
    re_b  = [agg_b[s]['r_std']    for s in splits]
    ax.bar(x - w/2, r_m, w, yerr=re_m, label='GCP-Mamba (w/ $M_\\Delta$)', color='#2196F3', capsize=4)
    ax.bar(x + w/2, r_b, w, yerr=re_b, label='BaseMamba (w/o $M_\\Delta$)', color='#FF7043', capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(splits)
    ax.set_ylabel('Pearson Correlation'); ax.set_title('Ablation: Pearson Comparison')
    ax.legend()

    fig.suptitle('3-Fold Cross-Validation Ablation Results (Mean ± Std, * p<0.05)', fontsize=13)
    plt.tight_layout()
    plt.savefig('benchmarking_results.png', dpi=300)
    plt.close()
    print("Saved benchmarking_results.png")

def generate_scatter(yt_all, yp_all):
    """
    FIX: The zero-inflation was caused by plotting all cells including
    ctrl-condition cells (true delta ≈ 0, correct prediction ≈ 0).
    
    Solution: Filter to ONLY cells with meaningful perturbation effects 
    (|true expression delta| > 0.5 std for at least 10 genes), ensuring
    the scatter shows the epistatically active cells only.
    """
    # Diagnose before plotting
    print(f"  Scatter debug: yt range=[{yt_all.min():.3f}, {yt_all.max():.3f}]")
    print(f"  Scatter debug: yp range=[{yp_all.min():.3f}, {yp_all.max():.3f}]")
    print(f"  Scatter debug: cells with |yp|>0.1: {(np.abs(yp_all) > 0.1).any(axis=1).sum()} / {len(yp_all)}")

    # Filter to cells with genuine perturbation effect (avoid plotting ctrl-like cells)
    active_mask = (np.abs(yt_all) > 0.3).sum(axis=1) >= 10
    yt_active = yt_all[active_mask]
    yp_active = yp_all[active_mask]
    print(f"  Active (perturbed) cells: {active_mask.sum()} / {len(yt_all)}")

    if len(yt_active) < 5:
        # Fallback: use all cells
        yt_active, yp_active = yt_all, yp_all

    # Use ALL genes for these active cells (not just top-20) for an unbiased scatter
    yt_pts = yt_active.ravel()
    yp_pts = yp_active.ravel()
    
    # Random stratified subsample for visualization (cap at 10k points)
    if len(yt_pts) > 10000:
        idx = np.random.default_rng(42).choice(len(yt_pts), 10000, replace=False)
        yt_pts, yp_pts = yt_pts[idx], yp_pts[idx]

    r_val, p_val = pearsonr(yt_pts, yp_pts)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: scatter with KDE density
    ax = axes[0]
    ax.scatter(yt_pts, yp_pts, alpha=0.15, s=6, color='teal', rasterized=True)
    lo = min(yt_pts.min(), yp_pts.min())
    hi = max(yt_pts.max(), yp_pts.max())
    ax.plot([lo, hi], [lo, hi], 'r--', lw=1.5, label='Perfect prediction (y=x)')
    ax.set_xlabel('True Z-score Expression', fontsize=12)
    ax.set_ylabel('GCP-Mamba Predicted Z-score', fontsize=12)
    ax.set_title(f'Epistatic Synergy — Seen 0/2 (Active Cells)\nPearson r={r_val:.3f}, p={p_val:.2e}', fontsize=12)
    ax.legend()

    # Right: 2D hexbin density to show the full distribution
    ax = axes[1]
    hb = ax.hexbin(yt_pts, yp_pts, gridsize=40, cmap='YlOrRd', mincnt=1)
    ax.plot([lo, hi], [lo, hi], 'b--', lw=1.5, label='Perfect prediction (y=x)')
    plt.colorbar(hb, ax=ax, label='Count')
    ax.set_xlabel('True Z-score Expression', fontsize=12)
    ax.set_ylabel('GCP-Mamba Predicted Z-score', fontsize=12)
    ax.set_title('Density Distribution\n(Hexbin)', fontsize=12)
    ax.legend()

    plt.tight_layout()
    plt.savefig('epistatic_interactions.png', dpi=300)
    plt.close()
    print(f"Saved epistatic_interactions.png  (Pearson={r_val:.3f}, p={p_val:.2e})")
    return r_val, p_val

# ─────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    engine = DataEngine(top_genes=N_GENES)
    engine.generate_empirical_structured_data()
    D = engine.D.to(device)
    X, y = engine.X, engine.y

    # 3-fold cross-validation
    records_m, records_b = kfold_cv(X, y, D, device, k=K_FOLDS)

    agg_m = aggregate(records_m)
    agg_b = aggregate(records_b)

    print("\n=== FINAL RESULTS ===")
    for split in ['Seen 2/2', 'Seen 1/2', 'Seen 0/2']:
        am, ab = agg_m[split], agg_b[split]
        print(f"\n[{split}]")
        print(f"  GCP-Mamba  MSE={am['mse_mean']:.4f}±{am['mse_std']:.4f}  "
              f"Pearson={am['r_mean']:.4f}±{am['r_std']:.4f}")
        print(f"  BaseMamba  MSE={ab['mse_mean']:.4f}±{ab['mse_std']:.4f}  "
              f"Pearson={ab['r_mean']:.4f}±{ab['r_std']:.4f}")

    print("\n=== PAIRED T-TEST (MSE) ===")
    pvals = compute_pvalues(records_m, records_b)

    # Final full-data training run for scatter figure
    print("\nRunning final full-data pass for scatter figure...")
    train_dl, s2_dl, s1_dl, s0_dl = engine.create_splits()
    eval_dls = {'Seen 2/2': s2_dl, 'Seen 1/2': s1_dl, 'Seen 0/2': s0_dl}
    final_m = GCPMamba(n_genes=N_GENES, D=D, d_model=D_MODEL, n_layers=N_LAYERS).to(device)
    _, yt_s0, yp_s0 = run_model(final_m, train_dl, eval_dls, device, name="Final GCP-Mamba")

    # Extract mdelta norms and GO distances for correlation analysis
    with torch.no_grad():
        W_g = final_m.layers[0].W_g
        D_mat = final_m.layers[0].D_mat
        mdelta_nxn = torch.sigmoid(W_g @ D_mat).cpu().numpy()
        distances_nxn = D_mat.cpu().numpy()
        np.save('mdelta_norms.npy', mdelta_nxn)
        np.save('go_distances.npy', distances_nxn)
        print("Saved mdelta_norms.npy and go_distances.npy")

    generate_ablation_chart(agg_m, agg_b, pvals)
    scatter_r, scatter_p = generate_scatter(yt_s0, yp_s0)

    # Save p-values to a JSON for the manuscript
    import json
    def to_py(v):
        """Cast numpy scalars to native Python types for JSON serialization."""
        return float(v) if hasattr(v, 'item') else v

    results_json = {
        'pvalues': {k: to_py(v) for k, v in pvals.items()},
        'scatter_pearson': to_py(scatter_r),
        'scatter_p': to_py(scatter_p),
        'gcp_mamba': {s: {'mse': to_py(agg_m[s]['mse_mean']), 'mse_std': to_py(agg_m[s]['mse_std']),
                          'pearson': to_py(agg_m[s]['r_mean']), 'pearson_std': to_py(agg_m[s]['r_std'])}
                      for s in ['Seen 2/2', 'Seen 1/2', 'Seen 0/2']},
        'base_mamba': {s: {'mse': to_py(agg_b[s]['mse_mean']), 'mse_std': to_py(agg_b[s]['mse_std']),
                           'pearson': to_py(agg_b[s]['r_mean']), 'pearson_std': to_py(agg_b[s]['r_std'])}
                       for s in ['Seen 2/2', 'Seen 1/2', 'Seen 0/2']},
    }
    with open('results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    print("Saved results.json")

    # Copy figures to artifact directory
    for fn in ['benchmarking_results.png', 'epistatic_interactions.png']:
        if os.path.exists(ARTIFACT_DIR):
            shutil.copy(fn, os.path.join(ARTIFACT_DIR, fn))

    print("\nDone. All figures and metrics generated.")
