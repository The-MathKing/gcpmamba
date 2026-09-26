import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import json

from model import GCPMamba, BaseMamba
from data_loader import DataEngine

# ─────────────────────────────────────────────────────────────────────────
# CONFIG — Frozen hyperparameters from canonical_results.py
# ─────────────────────────────────────────────────────────────────────────
N_GENES   = 500
D_MODEL   = 32
N_LAYERS  = 1
EPOCHS    = 5
LR        = 1e-3
SEEDS     = list(range(42, 42 + 3))
DEVICE    = "cpu"

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total = 0
    for x, y, conds in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / max(1, len(loader))

def evaluate_loader(model, loader, model_name, seed_idx, split_name, results_list):
    from scipy.stats import pearsonr
    model.eval()
    all_preds = []
    all_trues = []
    with torch.no_grad():
        for x, y, conds in loader:
            x = x.to(DEVICE)
            preds = model(x).cpu().numpy()
            trues = y.numpy()
            all_preds.append(preds)
            all_trues.append(trues)
            
    preds = np.concatenate(all_preds)
    trues = np.concatenate(all_trues)
    
    mse = np.mean((preds - trues) ** 2)
    p_flat = preds.flatten()
    t_flat = trues.flatten()
    if np.std(t_flat) > 1e-8 and np.std(p_flat) > 1e-8:
        r, _ = pearsonr(t_flat, p_flat)
    else:
        r = 0.0
        
    mse, r = float(mse), float(r)
    results_list.append({
        "Model": model_name,
        "Seed": seed_idx,
        "MSE": mse,
        "Pearson": r
    })
    return mse, r

def run_training_pipeline():
    print("Initializing DataEngine...")
    engine = DataEngine(
        top_genes=N_GENES,
        h5ad_path='data/replogle_rpe1_essential/perturb_processed.h5ad',
        splits_path='rpe1_splits.json'
    )
    engine.prepare_data()
    
    D = engine.D.to(DEVICE)
    
    # Create Permuted D
    D_permuted = D.clone()
    perm_idx = torch.randperm(N_GENES)
    D_permuted = D_permuted[perm_idx][:, perm_idx]
    
    results_list = []
    loss_history = {}
    
    for seed in SEEDS:
        print(f"\n=== Running Seed {seed} ===")
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Initialize Models
        models = {
            "BaseMamba": BaseMamba(n_genes=N_GENES, d_model=D_MODEL, n_layers=N_LAYERS).to(DEVICE),
            "GCP-Mamba": GCPMamba(n_genes=N_GENES, D=D, d_model=D_MODEL, n_layers=N_LAYERS).to(DEVICE),
            "GCP-Mamba (Permuted GO)": GCPMamba(n_genes=N_GENES, D=D_permuted, d_model=D_MODEL, n_layers=N_LAYERS).to(DEVICE)
        }
        
        loss_history[seed] = {}
        for name, model in models.items():
            print(f"  Training {name}...")
            optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
            criterion = nn.MSELoss()
            
            loss_history[seed][name] = []
            for ep in range(1, EPOCHS + 1):
                loss = train_one_epoch(model, engine.train_loader, optimizer, criterion)
                loss_history[seed][name].append(loss)
                if ep == EPOCHS or ep % 10 == 0:
                    print(f"    Epoch {ep:2d} | loss={loss:.4f}")
            
            # Evaluate all splits
            evaluate_loader(model, engine.test_singles_loader, name, seed, "Test Singles", results_list)
            
    with open("replogle_results.json", "w") as f:
        json.dump(results_list, f, indent=4)
        
    print("\n--- Final Metrics Saved to replogle_results.json ---")
    
    with open("training_losses.json", "w") as f:
        json.dump(loss_history, f)
    print("Saved training_losses.json.")
    
    # Summarize and do paired t-test
    df = pd.DataFrame(results_list)
    print("\n=== Replogle Test Singles Results ===")
    summary = df.groupby("Model")[["MSE", "Pearson"]].agg(["mean", "std"])
    print(summary)
    
    # Paired t-test
    gcp = df[df["Model"] == "GCP-Mamba"]["Pearson"].values
    perm = df[df["Model"] == "GCP-Mamba (Permuted GO)"]["Pearson"].values
    if len(gcp) > 0 and len(perm) > 0 and len(gcp) == len(perm):
        from scipy.stats import ttest_rel
        t_stat, p_val = ttest_rel(gcp, perm)
        print(f"\nPaired t-test (Pearson r): t={t_stat:.3f}, p={p_val:.4e}")
    else:
        print("Could not compute paired t-test")

if __name__ == "__main__":
    run_training_pipeline()
