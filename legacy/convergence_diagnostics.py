import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from model import BaseMamba
from data_loader import DataEngine

N_GENES = 500
D_MODEL = 32
N_LAYERS = 1
EPOCHS = 50
DEVICE = "cpu"
LR_SWEEP = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

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

def main():
    print("Initializing DataEngine...")
    engine = DataEngine(top_genes=N_GENES)
    engine.prepare_data()
    
    results = {}
    
    for lr in LR_SWEEP:
        print(f"\n=== Testing LR = {lr} ===")
        torch.manual_seed(42)
        np.random.seed(42)
        model = BaseMamba(n_genes=N_GENES, d_model=D_MODEL, n_layers=N_LAYERS).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.MSELoss()
        
        loss_hist = []
        for ep in range(1, EPOCHS + 1):
            loss = train_one_epoch(model, engine.train_loader, optimizer, criterion)
            loss_hist.append(loss)
            
        final_loss = np.mean(loss_hist[-5:])
        variance = np.var(loss_hist[-5:])
        
        results[str(lr)] = {
            "final_loss": float(final_loss),
            "variance": float(variance)
        }
        print(f"LR {lr}: Final Loss = {final_loss:.4f}, Variance = {variance:.4e}")
        
    with open('convergence_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Saved convergence_results.json")

if __name__ == "__main__":
    main()
