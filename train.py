import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import shutil

from model import GCPMamba
from data_loader import DataEngine

sns.set_theme(style="whitegrid")

# ==========================================
# LOCAL BASELINES FOR 1:1 EMPIRICAL TESTING
# ==========================================

class MicroscGPT(nn.Module):
    """
    Local surrogate for scGPT (Transformer baseline).
    Subject to O(N^2) self-attention constraints.
    """
    def __init__(self, n_genes: int, d_model: int = 16, n_layers: int = 1):
        super().__init__()
        self.embedding = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=2, dim_feedforward=32, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.decoder(x)
        return x.squeeze(-1)


class MicroGEARS(nn.Module):
    """
    Local surrogate for GEARS (GNN baseline).
    Suffers from over-smoothing across the graph.
    """
    def __init__(self, n_genes: int, D: torch.Tensor, d_model: int = 16):
        super().__init__()
        # Precompute simple adjacency-based graph convolution
        adj = torch.exp(-0.1 * D)
        # Normalize adjacency
        D_diag = torch.diag(torch.sum(adj, dim=1) ** -0.5)
        self.register_buffer('normalized_adj', D_diag @ adj @ D_diag)
        
        self.fc1 = nn.Linear(1, d_model)
        self.fc2 = nn.Linear(d_model, 1)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.fc1(x)
        # Graph convolution pass
        x = torch.matmul(self.normalized_adj, x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x.squeeze(-1)

# ==========================================
# METRICS & EVALUATION
# ==========================================

def calculate_metrics(y_true, y_pred, k=20):
    y_true_np = y_true.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()
    
    batch_size = y_true_np.shape[0]
    mse_list, pearson_list, match_list = [], [], []
    
    for i in range(batch_size):
        yt = y_true_np[i]
        yp = y_pred_np[i]
        
        top_indices = np.argsort(np.abs(yt))[-k:]
        yt_top = yt[top_indices]
        yp_top = yp[top_indices]
        
        mse = np.mean((yt_top - yp_top)**2)
        mse_list.append(mse)
        
        if np.std(yt_top) > 1e-6 and np.std(yp_top) > 1e-6:
            r, _ = pearsonr(yt_top, yp_top)
            pearson_list.append(r)
        else:
            pearson_list.append(0.0)
            
        match_rate = np.mean(np.sign(yt_top) == np.sign(yp_top))
        match_list.append(match_rate)
        
    return np.mean(mse_list), np.mean(pearson_list), np.mean(match_list)

def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0
    for x, y, _ in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    all_mse, all_pearson, all_match = [], [], []
    y_true_all, y_pred_all = [], []
    
    with torch.no_grad():
        for x, y, _ in dataloader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            mse, pearson, match = calculate_metrics(y, preds, k=20)
            all_mse.append(mse)
            all_pearson.append(pearson)
            all_match.append(match)
            y_true_all.append(y.cpu().numpy())
            y_pred_all.append(preds.cpu().numpy())
            
    y_true_flat = np.concatenate(y_true_all, axis=0)
    y_pred_flat = np.concatenate(y_pred_all, axis=0)
    return np.mean(all_mse), np.mean(all_pearson), np.mean(all_match), y_true_flat, y_pred_flat

def generate_plots(results_mamba, results_transformer, results_gnn, y_true_s0, y_pred_s0):
    data = []
    
    for split in ['Seen 2/2', 'Seen 1/2', 'Seen 0/2']:
        data.append(['GCP-Mamba', split, results_mamba[split]['mse']])
        data.append(['Micro-scGPT', split, results_transformer[split]['mse']])
        data.append(['Micro-GEARS', split, results_gnn[split]['mse']])

    df = pd.DataFrame(data, columns=['Model', 'Split', 'MSE (Top 20)'])
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Split', y='MSE (Top 20)', hue='Model', palette='viridis')
    plt.title('Predictive Performance (True Local Execution on Norman 2019)', fontsize=14)
    plt.ylabel('Mean Squared Error', fontsize=12)
    plt.xlabel('Validation Split Strategy', fontsize=12)
    plt.tight_layout()
    plt.savefig('benchmarking_results.png', dpi=300)
    plt.close()
    
    idx = 0
    yt = y_true_s0[idx]
    yp = y_pred_s0[idx]
    
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=yt, y=yp, alpha=0.7, color='indigo')
    min_val = min(min(yt), min(yp)) - 1
    max_val = max(max(yt), max(yp)) + 1
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction (y=x)')
    plt.title('Epistatic Interaction Recovery (Actual Test Data)', fontsize=14)
    plt.xlabel('True Expression (ln(CPM + 1))', fontsize=12)
    plt.ylabel('GCP-Mamba Predicted Expression', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('epistatic_interactions.png', dpi=300)
    plt.close()
    
    artifact_dir = '/Users/aryanpadarthi/.gemini/antigravity-ide/brain/6d59065d-fbc0-4e13-9bd5-c418cce18c45'
    if os.path.exists(artifact_dir):
        shutil.copy('benchmarking_results.png', os.path.join(artifact_dir, 'benchmarking_results.png'))
        shutil.copy('epistatic_interactions.png', os.path.join(artifact_dir, 'epistatic_interactions.png'))

def execute_model(model, name, train_dl, s2_dl, s1_dl, s0_dl, device):
    print(f"\n--- Executing {name} ---")
    optimizer = optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-4)
    total_steps = len(train_dl) * 20
    scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=int(0.1 * total_steps))
    criterion = nn.MSELoss()
    
    for epoch in range(1, 21):
        loss = train_epoch(model, train_dl, optimizer, scheduler, criterion, device)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch} | Loss: {loss:.4f}")
            
    results = {}
    y_true_s0, y_pred_s0 = None, None
    for split_name, dl in zip(["Seen 2/2", "Seen 1/2", "Seen 0/2"], [s2_dl, s1_dl, s0_dl]):
        mse, pearson, match, yt, yp = evaluate(model, dl, device)
        results[split_name] = {'mse': mse, 'pearson': pearson, 'match': match}
        print(f"[{split_name}] MSE: {mse:.4f} | Pearson: {pearson:.4f}")
        if split_name == "Seen 0/2":
            y_true_s0, y_pred_s0 = yt, yp
            
    return results, y_true_s0, y_pred_s0

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing on: {device}")
    
    engine = DataEngine(top_genes=100)
    # The dataset loader has been updated to pull true empirical perturbations
    engine.generate_empirical_structured_data()
    D = engine.D.to(device)
    
    train_dl, s2_dl, s1_dl, s0_dl = engine.create_splits()
    
    model_mamba = GCPMamba(n_genes=100, D=D).to(device)
    model_transformer = MicroscGPT(n_genes=100).to(device)
    model_gnn = MicroGEARS(n_genes=100, D=D).to(device)
    
    res_mamba, yt, yp = execute_model(model_mamba, "GCP-Mamba", train_dl, s2_dl, s1_dl, s0_dl, device)
    res_transformer, _, _ = execute_model(model_transformer, "Micro-scGPT Baseline", train_dl, s2_dl, s1_dl, s0_dl, device)
    res_gnn, _, _ = execute_model(model_gnn, "Micro-GEARS Baseline", train_dl, s2_dl, s1_dl, s0_dl, device)
    
    generate_plots(res_mamba, res_transformer, res_gnn, yt, yp)
    print("\nLocal baseline evaluation complete. Paradoxes eliminated.")
