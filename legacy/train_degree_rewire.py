import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import json

from model import GCPMamba
from data_loader import DataEngine

def degree_preserving_rewire(D, n_swaps=None):
    D_rewired = D.clone()
    N = D.shape[0]
    if n_swaps is None:
        n_swaps = N * 10

    for _ in range(n_swaps):
        i, j = np.random.randint(0, N, 2)
        k, l = np.random.randint(0, N, 2)
        if len({i, j, k, l}) == 4:
            D_rewired[i, j], D_rewired[k, l] = D_rewired[k, l].clone(), D_rewired[i, j].clone()
            D_rewired[j, i], D_rewired[l, k] = D_rewired[l, k].clone(), D_rewired[j, i].clone()

    return D_rewired

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total = 0
    n = 0
    for x, y, conds in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = nn.MSELoss()(model(x), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
        n += 1
    return total / max(1, n)

def evaluate(model, loader, device):
    from scipy.stats import pearsonr
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for x, y, conds in loader:
            x = x.to(device)
            all_preds.append(model(x).cpu().numpy())
            all_trues.append(y.numpy())
    preds = np.concatenate(all_preds)
    trues = np.concatenate(all_trues)
    mse = np.mean((preds - trues) ** 2)
    p_flat = preds.flatten()
    t_flat = trues.flatten()
    if np.std(t_flat) > 1e-8 and np.std(p_flat) > 1e-8:
        r, _ = pearsonr(t_flat, p_flat)
    else:
        r = 0.0
    return mse, r

def main():
    engine = DataEngine(top_genes=500, h5ad_path="data/norman/perturb_processed.h5ad", splits_path="splits/splits_manifest_0.json")
    engine.prepare_data()
    D = engine.D.to("cpu")
    D_rewired = degree_preserving_rewire(D)
    
    results = []
    
    for seed in [42, 43, 44]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = GCPMamba(n_genes=500, D=D_rewired, d_model=32, n_layers=1).to("cpu")
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        
        for ep in range(50):
            train_one_epoch(model, engine.train_loader, optimizer, "cpu")
            
        mse, r = evaluate(model, engine.seen1_loader, "cpu")
        results.append({"seed": seed, "MSE": mse, "Pearson": r})
        print(f"Seed {seed}: MSE={mse:.6f}, r={r:.4f}")
        
    df = pd.DataFrame(results)
    print("\n=== Degree-Preserving Rewire ===")
    print(df.agg(["mean", "std"]))
    
if __name__ == "__main__":
    main()
