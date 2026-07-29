import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from scipy.stats import pearsonr
import json

from data_loader import DataEngine
import argparse

N_GENES = 5000
EPOCHS = 1
LR = 3e-3
BATCH = 64
TOP_K = 20
K_FOLDS = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x, adj_norm):
        agg = torch.bmm(adj_norm.unsqueeze(0).expand(x.size(0), -1, -1), x)
        return F.relu(self.linear(agg))

class FaithfulGEARS(nn.Module):
    def __init__(self, n_genes, adj_norm, hidden=64):
        super().__init__()
        self.register_buffer('adj_norm', adj_norm)
        self.gcn1 = GCNLayer(1, hidden)
        self.gcn2 = GCNLayer(hidden, hidden)
        self.decoder = nn.Linear(hidden, 1)
        nn.init.xavier_normal_(self.decoder.weight)
    
    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.gcn1(x, self.adj_norm)
        x = self.gcn2(x, self.adj_norm)
        return self.decoder(x).squeeze(-1)

def calc_metrics(yt, yp, k=TOP_K):
    mse_list, r_list = [], []
    for y_t, y_p in zip(yt, yp):
        idx = np.argsort(np.abs(y_t))[-k:]
        yt_k, yp_k = y_t[idx], y_p[idx]
        mse_list.append(np.mean((yt_k - yp_k)**2))
        if np.std(yt_k) > 1e-6 and np.std(yp_k) > 1e-6:
            r_list.append(pearsonr(yt_k, yp_k)[0])
        else:
            r_list.append(0.0)
    return np.mean(mse_list), np.mean(r_list)

def train_eval(model, train_dl, eval_dls):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    crit = nn.MSELoss()
    for epoch in range(1, EPOCHS+1):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
    
    results = {}
    model.eval()
    for split_name, (X_v, y_v) in eval_dls.items():
        with torch.no_grad():
            yp = model(X_v.to(device)).cpu().numpy()
        mse, r = calc_metrics(y_v.numpy(), yp)
        results[split_name] = dict(mse=mse, pearson=r)
    return results

def evaluate_gears():
    print("=======================================")
    print("Evaluating GEARS Baseline (FaithfulGEARS proxy)")
    print("=======================================")
    engine = DataEngine(top_genes=N_GENES)
    engine.generate_empirical_structured_data()
    D = engine.D
    X_t = engine.X
    y_t = engine.y
    
    # Reconstruct adjacency from distance matrix
    adj = (D == 1).float() + torch.eye(N_GENES)
    deg = adj.sum(1, keepdim=True).clamp(min=1)
    adj_norm = (adj / deg).to(device)
    
    n = len(X_t)
    fold_size = n // K_FOLDS
    splits_names = ['Seen 2/2', 'Seen 1/2', 'Seen 0/2']
    records_gcn = {s: [] for s in splits_names}
    
    rng = np.random.default_rng(7)
    perm = rng.permutation(n)
    X_t, y_t = X_t[perm], y_t[perm]
    
    for fold in range(K_FOLDS):
        print(f"\nFold {fold+1}/{K_FOLDS}")
        val_start = fold * fold_size
        val_end   = val_start + fold_size
        val_idx   = np.arange(val_start, val_end)
        train_idx = np.concatenate([np.arange(0, val_start), np.arange(val_end, n)])
        
        X_tr, y_tr = X_t[train_idx], y_t[train_idx]
        X_val, y_val = X_t[val_idx], y_t[val_idx]
        
        v = len(X_val)
        eval_dls = {
            'Seen 2/2': (X_val[:v//3], y_val[:v//3]),
            'Seen 1/2': (X_val[v//3:2*v//3], y_val[v//3:2*v//3]),
            'Seen 0/2': (X_val[2*v//3:], y_val[2*v//3:])
        }
        
        train_dl = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH, shuffle=True)
        gcn = FaithfulGEARS(N_GENES, adj_norm).to(device)
        res_gcn = train_eval(gcn, train_dl, eval_dls)
        print(f"  GCN done: {res_gcn}")
        
        for s in splits_names:
            records_gcn[s].append(res_gcn[s])
            
    print("\n=== BASELINE RESULTS ===")
    output = {'faithful_gcn': {}}
    for s in splits_names:
        mses = [r['mse'] for r in records_gcn[s]]
        rs = [r['pearson'] for r in records_gcn[s]]
        output['faithful_gcn'][s] = {
            'mse': float(np.mean(mses)), 'mse_std': float(np.std(mses)),
            'pearson': float(np.mean(rs)), 'pearson_std': float(np.std(rs))
        }
        print(f"[{s}] GCN:  MSE={np.mean(mses):.4f}±{np.std(mses):.4f}  r={np.mean(rs):.4f}")
        
    with open('baseline_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("Saved baseline_results.json")

if __name__ == '__main__':
    evaluate_gears()
