import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from data_loader import DataEngine

class GEARSLight(nn.Module):
    """
    Lightweight GNN baseline imitating GEARS conceptually.
    Uses an initial dense layer followed by cross-gene interactions.
    """
    def __init__(self, n_genes, D, d_model=32):
        super().__init__()
        self.n_genes = n_genes
        self.d_model = d_model
        
        # We simulate the GO graph using the D matrix thresholded
        self.adj = (D < 0.5).float() # Simple threshold to create adjacency
        self.adj.fill_diagonal_(1.0)
        
        self.gene_emb = nn.Embedding(n_genes, d_model)
        self.pert_proj = nn.Linear(1, d_model)
        
        self.gnn1 = nn.Linear(d_model, d_model)
        self.gnn2 = nn.Linear(d_model, d_model)
        
        self.out_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        # x is (B, N) perturbation multi-hot
        B = x.shape[0]
        # Base representation for all genes
        emb = self.gene_emb.weight.unsqueeze(0).repeat(B, 1, 1) # (B, N, D)
        
        # Add perturbation signal
        p = self.pert_proj(x.unsqueeze(-1)) # (B, N, D)
        h = emb + p
        
        # 1-hop GNN layer
        # A * H W
        A = self.adj.to(x.device).unsqueeze(0) # (1, N, N)
        h = torch.bmm(A.repeat(B, 1, 1), h)
        h = torch.relu(self.gnn1(h))
        
        # 2-hop GNN layer
        h = torch.bmm(A.repeat(B, 1, 1), h)
        h = torch.relu(self.gnn2(h))
        
        out = self.out_proj(h).squeeze(-1) # (B, N)
        return out

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total = 0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = nn.MSELoss()(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / max(1, len(loader))

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
    device = "cpu"
    engine = DataEngine(top_genes=500, h5ad_path="data/norman/perturb_processed.h5ad", splits_path="splits/splits_manifest_0.json")
    engine.prepare_data()
    D = engine.D.to(device)
    
    results = []
    
    for seed in [42, 43]:
        torch.manual_seed(seed)
        model = GEARSLight(n_genes=500, D=D).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        
        for ep in range(30):
            train_one_epoch(model, engine.train_loader, optimizer, device)
            
        mse, r = evaluate(model, engine.seen1_loader, device)
        results.append({"Seed": seed, "MSE": mse, "Pearson": r})
        print(f"GEARS (Light) Seed {seed}: MSE={mse:.6f}, r={r:.4f}")
        
if __name__ == "__main__":
    main()
