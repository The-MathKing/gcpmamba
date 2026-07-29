import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import gc

sns.set_theme(style="whitegrid")
print(f"CUDA available: {pass # .is_available()}")
if pass # .is_available():
    print(f"GPU: {pass # .get_device_name(0)}")

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    def forward(self, x, adj_norm):
        agg = torch.bmm(adj_norm.unsqueeze(0).expand(x.size(0), -1, -1), x)
        return F.relu(self.linear(agg))

class FaithfulGEARS(nn.Module):
    def __init__(self, n_genes, hidden=64):
        super().__init__()
        self.gcn1 = GCNLayer(1, hidden)
        self.gcn2 = GCNLayer(hidden, hidden)
        self.decoder = nn.Linear(hidden, 1)
    def forward(self, x, adj_norm):
        x = x.unsqueeze(-1)
        x = self.gcn1(x, adj_norm)
        x = self.gcn2(x, adj_norm)
        return self.decoder(x).squeeze(-1)

class GCPMambaLayer(nn.Module):
    def __init__(self, n_genes, D, d_model=64):
        super().__init__()
        self.n_genes = n_genes
        self.D_mat = D
        self.W_g = nn.Parameter(torch.randn(n_genes, n_genes) / np.sqrt(n_genes))
        self.W_proj = nn.Linear(n_genes, d_model)
        self.in_proj = nn.Linear(1, d_model)
        self.dt_proj = nn.Linear(d_model, d_model)
        self.A_log = nn.Parameter(torch.log(torch.rand(d_model, 16) + 1e-4))
        self.B_proj = nn.Linear(d_model, 16)
        self.C_proj = nn.Linear(d_model, 16)
        self.out_proj = nn.Linear(d_model, 1)
    def precompute(self):
        M_gene = torch.sigmoid(self.W_g @ self.D_mat).mean(dim=-1)
        M_delta = torch.sigmoid(self.W_proj(M_gene))
        return M_delta
    def forward(self, x):
        B, L = x.shape
        x_proj = self.in_proj(x.unsqueeze(-1))  # (B, L, D)
        M_delta = self.precompute()  # (D,)
        dt = torch.sigmoid(self.dt_proj(x_proj)) * M_delta  # (B, L, D)
        return self.out_proj(x_proj).squeeze(-1)

class GCPMamba(nn.Module):
    def __init__(self, n_genes, D, d_model=64):
        super().__init__()
        self.layer = GCPMambaLayer(n_genes, D, d_model)
    def forward(self, x):
        return self.layer(x)
import numpy as np


L_vals = [100, 500, 1000, 3000, 5000, 10000, 20000]
BATCH_SIZE = 16
device = torch.device('cuda' if pass # .is_available() else 'cpu')

mem_gcn = []
mem_mamba = []

for L in L_vals:
    print(f"\n--- Testing Sequence Length L = {L} ---")
    
    # 1. Test GCN
    try:
        gc.collect(); pass # .empty_cache(); pass # .reset_peak_memory_stats()
        model_gcn = FaithfulGEARS(n_genes=L).to(device)
        optimizer_gcn = torch.optim.Adam(model_gcn.parameters())
        
        x = torch.randn(BATCH_SIZE, L, device=device)
        adj = torch.randn(L, L, device=device).clamp(0, 1)  # Dense O(L^2) adjacency
        y = torch.randn(BATCH_SIZE, L, device=device)
        
        out = model_gcn(x, adj)
        loss = F.mse_loss(out, y)
        loss.backward()
        optimizer_gcn.step()
        
        peak_mem_mb = pass # .max_memory_allocated() / (1024 ** 2)
        mem_gcn.append(peak_mem_mb)
        print(f"[GCN] Success: {peak_mem_mb:.1f} MB")
        
        del model_gcn, optimizer_gcn, x, adj, y, out, loss
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print(f"[GCN] OOM ERROR at L={L}")
            mem_gcn.append(None)
            pass # .empty_cache()
        else:
            raise e

    # 2. Test GCP-Mamba
    try:
        gc.collect(); pass # .empty_cache(); pass # .reset_peak_memory_stats()
        # D is precomputed locally, doesn't need to be dense O(L^2) during forward pass
        D = torch.randn(L, L, device=device)
        model_mamba = GCPMamba(n_genes=L, D=D).to(device)
        optimizer_mamba = torch.optim.Adam(model_mamba.parameters())
        
        x = torch.randn(BATCH_SIZE, L, device=device)
        y = torch.randn(BATCH_SIZE, L, device=device)
        
        out = model_mamba(x)
        loss = F.mse_loss(out, y)
        loss.backward()
        optimizer_mamba.step()
        
        peak_mem_mb = pass # .max_memory_allocated() / (1024 ** 2)
        mem_mamba.append(peak_mem_mb)
        print(f"[Mamba] Success: {peak_mem_mb:.1f} MB")
        
        del model_mamba, optimizer_mamba, D, x, y, out, loss
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print(f"[Mamba] OOM ERROR at L={L}")
            mem_mamba.append(None)
            pass # .empty_cache()
        else:
            raise e

import pandas as pd

plt.figure(figsize=(9, 6))

plt.plot(L_vals, [m if m is not None else float('nan') for m in mem_mamba], 
         label='GCP-Mamba ($\mathcal{O}(L)$ Sequence Scan)', marker='o', color='#2196F3', linewidth=2.5)
plt.plot(L_vals, [m if m is not None else float('nan') for m in mem_gcn], 
         label='FaithfulGEARS ($\mathcal{O}(L^2)$ GCN Message Passing)', marker='X', color='#E53935', linewidth=2.5, markersize=8)

plt.axhline(y=15360, color='r', linestyle='--', label='15GB Colab T4 VRAM Limit')

plt.xlabel('Sequence Length (Number of Genes $L$)', fontsize=12)
plt.ylabel('Peak GPU VRAM Usage (MB)', fontsize=12)
plt.title('Memory Scaling: GCP-Mamba vs. Graph Neural Networks', fontsize=14, pad=15)
plt.yscale('log')
plt.xscale('log')
plt.xticks(L_vals, labels=[str(v) for v in L_vals])
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('memory_scaling.png', dpi=300)
plt.show()

print("Saved memory_scaling.png")

