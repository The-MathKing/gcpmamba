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

from model import GCPMamba, BaseMamba
from data_loader import DataEngine

sns.set_theme(style="whitegrid")

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
        
        # Sort by actual significant variation
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

def generate_plots(results_mamba, results_base, y_true_s0, y_pred_s0):
    data = []
    
    for split in ['Seen 2/2', 'Seen 1/2', 'Seen 0/2']:
        data.append(['GCP-Mamba (w/ G)', split, results_mamba[split]['mse']])
        data.append(['BaseMamba (w/o G)', split, results_base[split]['mse']])

    df = pd.DataFrame(data, columns=['Model', 'Split', 'MSE (Top 20)'])
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Split', y='MSE (Top 20)', hue='Model', palette='rocket')
    plt.title('Ablation Performance Analysis: Structural Graph Conditioning', fontsize=14)
    plt.ylabel('Mean Squared Error', fontsize=12)
    plt.xlabel('Validation Split Strategy', fontsize=12)
    plt.tight_layout()
    plt.savefig('benchmarking_results.png', dpi=300)
    plt.close()
    
    # Isolate highly synergistic cell interactions for scatter mapping
    idx = 0
    yt = y_true_s0[idx]
    yp = y_pred_s0[idx]
    
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=yt, y=yp, alpha=0.7, color='teal')
    min_val = min(min(yt), min(yp)) - 1
    max_val = max(max(yt), max(yp)) + 1
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction (y=x)')
    plt.title('Corrected Epistatic Synergy Recovery (Unseen 0/2 Test Data)', fontsize=14)
    plt.xlabel('True Expression (Empirical)', fontsize=12)
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
    print(f"\n--- Executing Ablation Target: {name} ---")
    optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4) # Higher learning rate for rapid local convergence
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
    engine.generate_empirical_structured_data()
    D = engine.D.to(device)
    
    train_dl, s2_dl, s1_dl, s0_dl = engine.create_splits()
    
    # Strict 1:1 Ablation Study (GCP-Mamba w/ G vs BaseMamba w/o G)
    model_mamba = GCPMamba(n_genes=100, D=D).to(device)
    model_base = BaseMamba(n_genes=100).to(device)
    
    res_mamba, yt, yp = execute_model(model_mamba, "GCP-Mamba (w/ Graph Topology)", train_dl, s2_dl, s1_dl, s0_dl, device)
    res_base, _, _ = execute_model(model_base, "BaseMamba (w/o Graph Topology)", train_dl, s2_dl, s1_dl, s0_dl, device)
    
    generate_plots(res_mamba, res_base, yt, yp)
    print("\nAblation study and figure generation complete.")
