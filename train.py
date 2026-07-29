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
EPOCHS    = 50
LR        = 1e-3
SEEDS     = [42, 100, 2026, 777, 999]
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
                        "True_Value": trues[i, g_idx],
                        "Pred_Value": preds[i, g_idx]
                    })

def run_training_pipeline():
    print("Initializing DataEngine...")
    engine = DataEngine(top_genes=N_GENES)
    engine.prepare_data()
    
    D = engine.D.to(DEVICE)
    
    # Create Permuted D
    D_permuted = D.clone()
    perm_idx = torch.randperm(N_GENES)
    D_permuted = D_permuted[perm_idx][:, perm_idx]
    
    results_list = []
    
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
        
        for name, model in models.items():
            print(f"  Training {name}...")
            optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
            criterion = nn.MSELoss()
            
            for ep in range(1, EPOCHS + 1):
                loss = train_one_epoch(model, engine.train_loader, optimizer, criterion)
                if ep == EPOCHS or ep % 10 == 0:
                    print(f"    Epoch {ep:2d} | loss={loss:.4f}")
            
            # Evaluate all splits
            evaluate_loader(model, engine.seen2_loader, name, seed, "Seen 2/2", results_list)
            evaluate_loader(model, engine.seen1_loader, name, seed, "Seen 1/2", results_list)
            evaluate_loader(model, engine.seen0_loader, name, seed, "Seen 0/2", results_list)
    
    # Save predictions
    df = pd.DataFrame(results_list)
    df.to_csv("predictions.csv", index=False)
    print("\nSaved predictions.csv successfully. All further analysis will use this canonical file.")

if __name__ == "__main__":
    run_training_pipeline()
