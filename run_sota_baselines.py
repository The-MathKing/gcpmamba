import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
from sklearn.linear_model import Ridge

from data_loader import DataEngine

N_GENES = 500
EPOCHS = 50
LR = 1e-3
SEEDS = [42, 100, 2026, 777, 999]
DEVICE = "cpu"

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x, adj_norm):
        agg = torch.bmm(adj_norm.unsqueeze(0).expand(x.size(0), -1, -1), x)
        return F.relu(self.linear(agg))

class FaithfulGEARS(nn.Module):
    def __init__(self, n_genes, adj_norm, hidden=32):
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

def evaluate_and_append(y_true, y_pred, conds, model_name, seed_idx, split_name, results_list):
    for i, cond in enumerate(conds):
        for g_idx in range(N_GENES):
            results_list.append({
                "Split": split_name,
                "Condition": cond,
                "Model": model_name,
                "Seed": seed_idx,
                "Gene_Idx": g_idx,
                "True_Value": y_true[i, g_idx],
                "Pred_Value": y_pred[i, g_idx]
            })

def run_baselines():
    print("Initializing DataEngine...")
    engine = DataEngine(top_genes=N_GENES)
    engine.prepare_data()
    
    D = engine.D
    adj = (D == 1).float() + torch.eye(N_GENES)
    deg = adj.sum(1, keepdim=True).clamp(min=1)
    adj_norm = (adj / deg).to(DEVICE)
    
    # Load existing predictions to append
    try:
        df_exist = pd.read_csv("predictions.csv")
        results_list = df_exist.to_dict('records')
        print(f"Loaded existing predictions.csv with {len(results_list)} records.")
    except:
        results_list = []
        print("Starting fresh predictions list.")
        
    # Pre-calculate condition singles for additive baseline
    # Extract training singles
    train_X, train_y, train_c = [], [], []
    for x, y, c in engine.train_loader:
        train_X.append(x.numpy())
        train_y.append(y.numpy())
        train_c.extend(c)
    train_X = np.concatenate(train_X)
    train_y = np.concatenate(train_y)
    
    single_effects = {}
    for c_name, y_val in zip(train_c, train_y):
        if '+' not in c_name:
            # wait, they end with +ctrl
            if c_name.endswith('+ctrl'):
                single_effects[c_name.split('+')[0]] = y_val

    cond_mean = train_y.mean(axis=0)
    
    for seed in SEEDS:
        print(f"\n=== Baseline Seed {seed} ===")
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # 1. GEARS (GCN)
        gcn = FaithfulGEARS(N_GENES, adj_norm).to(DEVICE)
        opt = torch.optim.AdamW(gcn.parameters(), lr=LR, weight_decay=1e-4)
        crit = nn.MSELoss()
        
        for ep in range(1, EPOCHS+1):
            gcn.train()
            for xb, yb, _ in engine.train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                loss = crit(gcn(xb), yb)
                loss.backward()
                opt.step()
        
        # 2. Linear Regression (Ridge)
        ridge = Ridge(alpha=1.0)
        ridge.fit(train_X, train_y)
        
        # Eval loop
        splits = [
            ("Seen 2/2", engine.seen2_loader),
            ("Seen 1/2", engine.seen1_loader),
            ("Seen 0/2", engine.seen0_loader)
        ]
        
        gcn.eval()
        with torch.no_grad():
            for split_name, loader in splits:
                if len(loader.dataset) == 0: continue
                
                for xb, yb, conds in loader:
                    y_true = yb.numpy()
                    
                    # GEARS
                    y_gcn = gcn(xb.to(DEVICE)).cpu().numpy()
                    evaluate_and_append(y_true, y_gcn, conds, "GEARS", seed, split_name, results_list)
                    
                    # Linear
                    y_lin = ridge.predict(xb.numpy())
                    evaluate_and_append(y_true, y_lin, conds, "Linear", seed, split_name, results_list)
                    
                    # Condition Mean
                    y_mean = np.tile(cond_mean, (len(conds), 1))
                    evaluate_and_append(y_true, y_mean, conds, "Condition Mean", seed, split_name, results_list)
                    
                    # Additive
                    y_add = np.zeros_like(y_true)
                    for i, cond in enumerate(conds):
                        g1, g2 = cond.split('+')
                        eff1 = single_effects.get(g1, np.zeros(N_GENES))
                        eff2 = single_effects.get(g2, np.zeros(N_GENES))
                        y_add[i] = eff1 + eff2
                    evaluate_and_append(y_true, y_add, conds, "Additive", seed, split_name, results_list)

    df = pd.DataFrame(results_list)
    df.to_csv("predictions.csv", index=False)
    print("\nSaved all baselines to predictions.csv successfully.")

if __name__ == "__main__":
    run_baselines()
