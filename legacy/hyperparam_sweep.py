import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime

from model import GCPMamba
from data_loader import DataEngine

CONFIG = {
    "n_genes": 500,
    "n_layers": 1,
    "epochs": 50,
    "lr": 1e-3,
    "seeds": [42, 43],  # 2 seeds
    "device": "cpu",
    "d_models": [16, 32, 64],
    "h5ad_path": "data/norman/perturb_processed.h5ad",
    "splits_path": "splits/splits_manifest_0.json"
}

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total = 0
    for x, y, conds in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = nn.MSELoss()(model(x), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / max(1, len(loader))

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
    device = CONFIG["device"]
    
    print("Initializing DataEngine for Hyperparameter Sweep...")
    engine = DataEngine(
        top_genes=CONFIG["n_genes"],
        h5ad_path=CONFIG["h5ad_path"],
        splits_path=CONFIG["splits_path"]
    )
    engine.prepare_data()
    D = engine.D.to(device)
    
    results = []
    
    for d_model in CONFIG["d_models"]:
        print(f"\n--- Testing d_model = {d_model} ---")
        for seed in CONFIG["seeds"]:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            model = GCPMamba(
                n_genes=CONFIG["n_genes"], D=D,
                d_model=d_model, n_layers=CONFIG["n_layers"]
            ).to(device)
            
            optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-4)
            
            print(f"  Training Seed {seed}...")
            for ep in range(CONFIG["epochs"]):
                train_one_epoch(model, engine.train_loader, optimizer, device)
                
            mse, r = evaluate(model, engine.seen1_loader, device)
            print(f"    Seed {seed}: MSE={mse:.6f}, r={r:.4f}")
            
            results.append({
                "d_model": d_model,
                "seed": seed,
                "MSE": mse,
                "Pearson": r
            })
            
    df = pd.DataFrame(results)
    print("\n=== Hyperparameter Sweep Results ===")
    summary = df.groupby("d_model")[["MSE", "Pearson"]].agg(["mean", "std"])
    print(summary)
    
    df.to_csv("hyperparam_sweep_results.csv", index=False)
    
if __name__ == "__main__":
    main()
