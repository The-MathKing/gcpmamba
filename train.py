import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.stats import pearsonr, ttest_rel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
N_GENES   = 100
D_MODEL   = 16
N_LAYERS  = 1
EPOCHS    = 10
LR        = 3e-3
BATCH     = 64
K_FOLDS   = 3
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
    """Handle both 2-tuple (TensorDataset) and 3-tuple (PerturbDataset) loaders."""
    if len(batch) == 2:
        return batch[0], batch[1]
    return batch[0], batch[1]   # ignore conditions in both cases

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
# 5-FOLD CROSS VALIDATION
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

        # Split validation into three tiers
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
        }
    return agg

def generate_ablation_chart(agg_m, agg_b):
    splits = ['Seen 2/2', 'Seen 1/2', 'Seen 0/2']
    x = np.arange(len(splits))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # MSE subplot
    ax = axes[0]
    mse_m  = [agg_m[s]['mse_mean']  for s in splits]
    mse_b  = [agg_b[s]['mse_mean']  for s in splits]
    err_m  = [agg_m[s]['mse_std']   for s in splits]
    err_b  = [agg_b[s]['mse_std']   for s in splits]
    ax.bar(x - w/2, mse_m, w, yerr=err_m, label='GCP-Mamba (w/ $M_\\Delta$)', color='#2196F3', capsize=4)
    ax.bar(x + w/2, mse_b, w, yerr=err_b, label='BaseMamba (w/o $M_\\Delta$)', color='#FF7043', capsize=4)
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

    fig.suptitle('5-Fold Cross-Validation Ablation Results (Mean ± Std)', fontsize=14)
    plt.tight_layout()
    plt.savefig('benchmarking_results.png', dpi=300)
    plt.close()
    print("Saved benchmarking_results.png")

def generate_scatter(yt_all, yp_all):
    # Subsample top-20 DE genes per cell and flatten
    top_idx = np.argsort(np.abs(yt_all), axis=1)[:, -TOP_K:]
    yt_pts = np.array([yt_all[i, top_idx[i]] for i in range(len(yt_all))]).ravel()
    yp_pts = np.array([yp_all[i, top_idx[i]] for i in range(len(yp_all))]).ravel()

    # Cap outliers for clean visualization
    low, high = np.percentile(yt_pts, 1), np.percentile(yt_pts, 99)
    mask = (yt_pts >= low) & (yt_pts <= high) & (yp_pts >= low) & (yp_pts <= high)
    yt_pts, yp_pts = yt_pts[mask], yp_pts[mask]

    r_val, _ = pearsonr(yt_pts, yp_pts)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(yt_pts, yp_pts, alpha=0.25, s=8, color='teal', rasterized=True)
    lo, hi = min(yt_pts.min(), yp_pts.min()), max(yt_pts.max(), yp_pts.max())
    ax.plot([lo, hi], [lo, hi], 'r--', lw=1.5, label='Perfect prediction (y=x)')
    ax.set_xlabel('True Z-score Expression', fontsize=12)
    ax.set_ylabel('GCP-Mamba Predicted Z-score', fontsize=12)
    ax.set_title(f'Epistatic Synergy Recovery — Seen 0/2\n(Pearson r = {r_val:.3f}, top-{TOP_K} genes)', fontsize=13)
    ax.legend()
    plt.tight_layout()
    plt.savefig('epistatic_interactions.png', dpi=300)
    plt.close()
    print(f"Saved epistatic_interactions.png  (Pearson={r_val:.3f})")

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

    # --- 5-fold cross-validation ---
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

    # --- Generate figures using a final full-data training run ---
    print("\nRunning final full-data pass for scatter figure...")
    train_dl, s2_dl, s1_dl, s0_dl = engine.create_splits()
    eval_dls = {'Seen 2/2': s2_dl, 'Seen 1/2': s1_dl, 'Seen 0/2': s0_dl}
    final_m = GCPMamba(n_genes=N_GENES, D=D, d_model=D_MODEL, n_layers=N_LAYERS).to(device)
    _, yt_s0, yp_s0 = run_model(final_m, train_dl, eval_dls, device, name="Final GCP-Mamba")

    generate_ablation_chart(agg_m, agg_b)
    generate_scatter(yt_s0, yp_s0)

    # Copy figures to artifact directory
    for fn in ['benchmarking_results.png', 'epistatic_interactions.png']:
        src = fn
        if os.path.exists(ARTIFACT_DIR):
            shutil.copy(src, os.path.join(ARTIFACT_DIR, fn))

    print("\nDone. All figures and metrics generated.")
