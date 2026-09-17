import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from data_loader import DataEngine
from model import GCPMamba

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total = 0
    n = 0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = nn.MSELoss()(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item()
        n += 1
    return total / max(1, n)

def evaluate(model, loader, device):
    from scipy.stats import pearsonr
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            all_preds.append(model(x.to(device)).cpu().numpy())
            all_trues.append(y.numpy())
    preds = np.concatenate(all_preds)
    trues = np.concatenate(all_trues)
    mse = np.mean((preds - trues) ** 2)
    t_flat = trues.flatten()
    p_flat = preds.flatten()
    r = pearsonr(t_flat, p_flat)[0] if np.std(t_flat)>0 and np.std(p_flat)>0 else 0
    return mse, r

def main():
    print("Running light test...")
    engine = DataEngine(top_genes=500, h5ad_path="data/norman/perturb_processed.h5ad", splits_path="splits/splits_manifest_0.json")
    engine.prepare_data()
    D = engine.D.to("cpu")
    
    # Verify input vectors are not zero
    print("Checking input vector sparsity...")
    zero_inputs = sum([1 for x, y, c in engine.train_loader.dataset if torch.all(x == 0).item()])
    total_inputs = len(engine.train_loader.dataset)
    print(f"{zero_inputs}/{total_inputs} training samples have all-zero input vectors.")
    
    torch.manual_seed(42)
    model = GCPMamba(n_genes=500, D=D, d_model=32, n_layers=1).to("cpu")
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    print("Training...")
    for ep in range(10):
        loss = train_one_epoch(model, engine.train_loader, optimizer, "cpu")
        print(f"Epoch {ep+1}: Loss = {loss:.4f}")
        
    mse, r = evaluate(model, engine.seen1_loader, "cpu")
    print(f"Final Test (Seen 1/2): MSE={mse:.6f}, r={r:.4f}")

if __name__ == "__main__":
    main()
