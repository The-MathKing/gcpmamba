import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import scanpy as sc
import os

from model import GCPMamba

class DataEngineDecoupled:
    def __init__(self, top_genes=500, h5ad_path="data/norman/perturb_processed.h5ad"):
        self.top_genes = top_genes
        self.h5ad_path = h5ad_path
        
    def prepare_data(self):
        adata = sc.read_h5ad(self.h5ad_path)
        
        # Get all unique perturbed genes
        unique_conditions = adata.obs['condition'].unique()
        pert_genes = set()
        for cond in unique_conditions:
            if cond == 'ctrl': continue
            for g in cond.split('+'):
                if g != 'ctrl':
                    pert_genes.add(g)
        self.all_pert_genes = sorted(list(pert_genes))
        self.pert_to_idx = {g: i+1 for i, g in enumerate(self.all_pert_genes)} # 0 is ctrl/pad
        
        # Keep 500 HVGs
        adata_train = adata[~adata.obs['condition'].isin(['ctrl'])].copy()
        sc.pp.filter_genes(adata_train, min_cells=3)
        sc.pp.normalize_total(adata_train, target_sum=1e4)
        sc.pp.log1p(adata_train)
        sc.pp.highly_variable_genes(adata_train, n_top_genes=self.top_genes, subset=True)
        
        self.hugo_names = adata_train.var['gene_name'].tolist()
        self.gene_names = adata_train.var_names.tolist()
        
        adata = adata[:, self.gene_names].copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        
        X_base = adata.X.toarray() if hasattr(adata.X, "toarray") else np.array(adata.X)
        self.X_mean = X_base.mean(axis=0)
        self.X_std = X_base.std(axis=0) + 1e-6
        X_base_z = (X_base - self.X_mean) / self.X_std
        
        ctrl_mask = adata.obs['condition'] == 'ctrl'
        ctrl_mean = X_base_z[ctrl_mask].mean(axis=0)
        
        # Dummy D for now (identity)
        self.D = torch.eye(self.top_genes)
        
        cond_X_list = []
        cond_y_list = []
        cond_names = []
        
        for cond in unique_conditions:
            if cond == 'ctrl': continue
            mask = adata.obs['condition'] == cond
            cond_mean = X_base_z[mask].mean(axis=0)
            delta_y = cond_mean - ctrl_mean
            
            p_idx = [0, 0]
            genes = [g for g in cond.split('+') if g != 'ctrl']
            for i, g in enumerate(genes):
                if i < 2 and g in self.pert_to_idx:
                    p_idx[i] = self.pert_to_idx[g]
                    
            cond_X_list.append(p_idx)
            cond_y_list.append(delta_y)
            cond_names.append(cond)
            
        self.X_tensor = torch.tensor(cond_X_list, dtype=torch.long)
        self.y_tensor = torch.tensor(np.array(cond_y_list), dtype=torch.float32)
        
        # Simple split: 80% train, 20% test
        N = len(cond_names)
        idx = np.random.permutation(N)
        train_idx, test_idx = idx[:int(0.8*N)], idx[int(0.8*N):]
        
        self.train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self.X_tensor[train_idx], self.y_tensor[train_idx]),
            batch_size=32, shuffle=True
        )
        self.test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self.X_tensor[test_idx], self.y_tensor[test_idx]),
            batch_size=32, shuffle=False
        )

class DecoupledMamba(nn.Module):
    def __init__(self, n_genes=500, n_pert_vocab=200, d_model=32, D=None):
        super().__init__()
        self.pert_emb = nn.Embedding(n_pert_vocab, d_model)
        self.gene_base = nn.Parameter(torch.randn(1, n_genes, d_model))
        self.mamba = GCPMamba(n_genes=n_genes, D=D, d_model=d_model, n_layers=2)
        
    def forward(self, pert_idx):
        # pert_idx: (B, 2)
        B = pert_idx.shape[0]
        # Sum embeddings for up to 2 perturbations
        p = self.pert_emb(pert_idx[:, 0]) + self.pert_emb(pert_idx[:, 1]) # (B, d_model)
        
        # Base sequence
        x = self.gene_base.expand(B, -1, -1) + p.unsqueeze(1) # (B, n_genes, d_model)
        
        # Bypass embedding layer in GCPMamba
        # Normally GCPMamba expects (B, L) and applies self.embedding
        # We will directly feed to the layers
        for layer in self.mamba.layers:
            x = layer(x)
        out = self.mamba.decoder(self.mamba.norm_f(x)).squeeze(-1)
        return out

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total = 0
    n = 0
    for x, y in loader:
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
        for x, y in loader:
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
    print("Running Decoupled Mamba test...")
    engine = DataEngineDecoupled()
    engine.prepare_data()
    D = engine.D.to("cpu")
    
    n_pert_vocab = len(engine.all_pert_genes) + 1
    print(f"Total perturbed gene vocabulary size: {n_pert_vocab}")
    
    torch.manual_seed(42)
    model = DecoupledMamba(n_genes=500, n_pert_vocab=n_pert_vocab, d_model=32, D=D).to("cpu")
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    for ep in range(15):
        loss = train_one_epoch(model, engine.train_loader, optimizer, "cpu")
        mse, r = evaluate(model, engine.test_loader, "cpu")
        print(f"Epoch {ep+1:2d}: Train Loss={loss:.4f} | Test MSE={mse:.4f}, Test r={r:.4f}")

if __name__ == "__main__":
    main()
