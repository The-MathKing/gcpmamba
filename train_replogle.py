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
# CONFIG
# ─────────────────────────────────────────────────────────────────────────
N_GENES   = 500
D_MODEL   = 32
N_LAYERS  = 1
EPOCHS = 20
LR        = 1e-3
SEEDS     = [42]
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
    model.eval()
    with torch.no_grad():
        for x, y, conds in loader:
            x = x.to(DEVICE)
            preds = model(x).cpu().numpy()
            trues = y.numpy()
            
            for i, cond in enumerate(conds):
                for g_idx in range(N_GENES):
                    # Only log top genes to save space or just log all.
                    # Since N_GENES=500, we have 500 rows per condition per model.
                    results_list.append({
                        "Split": split_name,
                        "Condition": cond,
                        "Model": model_name,
                        "Seed": seed_idx,
                        "Gene_Idx": g_idx,
                        "True_Value": float(trues[i, g_idx]),
                        "Pred_Value": float(preds[i, g_idx])
                    })

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
    
    # Compute M_delta vs GO distance correlation for GCP-Mamba
    print("\n--- M_delta vs GO distance analysis ---")
    d_flat = D.cpu().numpy().flatten()
    m_deltas = []
    # Using the last seed's GCP-Mamba model
    model = models["GCP-Mamba"]
    model.eval()
    with torch.no_grad():
        W_g = model.layers[0].W_g.cpu()
        D_cpu = D.cpu()
        M_gene = torch.sigmoid(W_g @ D_cpu).mean(dim=-1) # (N,)
        # Create pairwise matrix from M_gene
        M_pair = M_gene.unsqueeze(1).repeat(1, N_GENES)
        m_flat = M_pair.numpy().flatten()
        
    from scipy.stats import pearsonr
    valid = (d_flat > 0)
    r_val, p_val = pearsonr(d_flat[valid], m_flat[valid])
    print(f"M_delta vs Distance: r = {r_val:.3f}, n = {valid.sum()}, p = {p_val:.4e}")
    with open("m_delta_stats.json", "w") as f:
        json.dump({"r": float(r_val), "n": int(valid.sum()), "p": float(p_val)}, f)

if __name__ == "__main__":
    run_training_pipeline()
