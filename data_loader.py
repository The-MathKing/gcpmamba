import numpy as np
import networkx as nx
from typing import Tuple, List
import torch
from torch.utils.data import Dataset, DataLoader

class PerturbDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor, condition: List[str]):
        self.X = X
        self.y = y
        self.condition = condition
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.condition[idx]

class DataEngine:
    def __init__(self, top_genes: int = 100):
        self.top_genes = top_genes
        self.D = None
        self.X = None
        self.y = None
        self.conditions = None
        self.y_mean = None
        self.y_std = None
        
    def generate_empirical_structured_data(self):
        """
        Loads empirical PBMC3k data and applies:
        1. Z-score standardization to BOTH inputs and targets (fixes MSE ~11 and the y=0 flatline)
        2. Orthogonal noise for target perturbation (severs circular GO leakage)
        3. Empirical covariance-based structural graph D
        """
        print("Initializing empirical dataset mapping...")
        import scanpy as sc
        
        adata = sc.datasets.pbmc3k()
        sc.pp.filter_genes(adata, min_cells=3)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=self.top_genes, subset=True)
        X_base = adata.X.toarray() if hasattr(adata.X, "toarray") else np.array(adata.X)
        
        n_cells, n_genes = X_base.shape
        
        # --- CRITICAL FIX: Z-score standardize inputs ---
        X_mean = X_base.mean(axis=0, keepdims=True)
        X_std = X_base.std(axis=0, keepdims=True) + 1e-8
        X_base_z = (X_base - X_mean) / X_std
        
        # Build empirical topological graph D from correlation structure
        cov_matrix = np.corrcoef(X_base.T)
        cov_matrix = np.nan_to_num(cov_matrix)
        adj_matrix = (np.abs(cov_matrix) > 0.3).astype(float)
        G = nx.from_numpy_array(adj_matrix)
        
        length_dict = dict(nx.all_pairs_shortest_path_length(G))
        D = np.zeros((self.top_genes, self.top_genes))
        for i in range(self.top_genes):
            for j in range(self.top_genes):
                D[i, j] = length_dict.get(i, {}).get(j, 10)
        self.D = torch.tensor(D, dtype=torch.float32)
        print("Structural graph extracted from empirical covariance.")

        # Decouple target generation from graph topology using an orthogonal drift
        rng = np.random.default_rng(seed=42)
        orthogonal_drift = rng.normal(0, 1, (n_genes, n_genes))

        y_target = np.copy(X_base_z)
        conditions = []
        
        for i in range(n_cells):
            cond_type = rng.choice(['ctrl', 'single', 'double'], p=[0.2, 0.4, 0.4])
            
            if cond_type == 'ctrl':
                conditions.append('ctrl')
            elif cond_type == 'single':
                g1 = rng.integers(0, n_genes)
                conditions.append(f"gene_{g1}")
                effect = orthogonal_drift[g1, :] * rng.normal(0.3, 0.1, n_genes)
                y_target[i, :] += effect
            else:
                g1, g2 = rng.choice(n_genes, 2, replace=False)
                conditions.append(f"gene_{g1}+gene_{g2}")
                effect1 = orthogonal_drift[g1, :] * rng.normal(0.3, 0.1, n_genes)
                effect2 = orthogonal_drift[g2, :] * rng.normal(0.3, 0.1, n_genes)
                # True epistatic synergy independent of graph topology
                synergy = (orthogonal_drift[g1, :] * orthogonal_drift[g2, :]) * rng.normal(0.5, 0.15, n_genes)
                y_target[i, :] += effect1 + effect2 + synergy
        
        # --- CRITICAL FIX: Z-score standardize targets (fixes MSE >> 1 and y=0 flatline) ---
        self.y_mean = y_target.mean(axis=0, keepdims=True)
        self.y_std = y_target.std(axis=0, keepdims=True) + 1e-8
        y_target_z = (y_target - self.y_mean) / self.y_std
        
        self.X = torch.tensor(X_base_z, dtype=torch.float32)
        self.y = torch.tensor(y_target_z, dtype=torch.float32)
        self.conditions = conditions
        print(f"Dataset ready: {n_cells} cells, {n_genes} genes.")
        print(f"  Input  range: [{self.X.min():.2f}, {self.X.max():.2f}]")
        print(f"  Target range: [{self.y.min():.2f}, {self.y.max():.2f}]")

    def create_splits(self) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
        n_cells = len(self.X)
        rng = np.random.default_rng(seed=99)
        indices = rng.permutation(n_cells)
        
        split1 = int(0.60 * n_cells)
        split2 = int(0.75 * n_cells)
        split3 = int(0.90 * n_cells)
        
        train_idx = indices[:split1]
        seen2_idx = indices[split1:split2]
        seen1_idx = indices[split2:split3]
        seen0_idx = indices[split3:]
        
        def get_loader(idx, shuffle=False):
            ds = PerturbDataset(
                self.X[idx], self.y[idx],
                [self.conditions[i] for i in idx]
            )
            return DataLoader(ds, batch_size=64, shuffle=shuffle)
            
        return (get_loader(train_idx, True),
                get_loader(seen2_idx),
                get_loader(seen1_idx),
                get_loader(seen0_idx))
