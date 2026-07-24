import numpy as np
import pandas as pd
import networkx as nx
from typing import Tuple, List
import torch
from torch.utils.data import Dataset, DataLoader
import os

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
        
    def generate_empirical_structured_data(self):
        """
        Uses empirical scanpy data but generates combinatorial targets using a 
        completely decoupled biological drift matrix to absolutely eliminate circular leakage.
        """
        print("Initializing empirical dataset mapping...")
        import scanpy as sc
        
        # Load empirical baseline states
        adata = sc.datasets.pbmc3k()
        sc.pp.filter_genes(adata, min_cells=3)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=self.top_genes, subset=True)
        X_base = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        
        n_cells = X_base.shape[0]
        
        # Build empirical topological graph D
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
        
        # To COMPLETELY ELIMINATE CIRCULAR LEAKAGE, we generate the targets using a 
        # completely orthogonal, decoupled biological signature mapping. 
        # The targets DO NOT use D or cov_matrix.
        orthogonal_biological_drift = np.random.normal(0, 1, (self.top_genes, self.top_genes))
        
        X_perturbed = np.copy(X_base)
        y_target = np.copy(X_base)
        conditions = []
        
        for i in range(n_cells):
            cond_type = np.random.choice(['ctrl', 'single', 'double'], p=[0.2, 0.4, 0.4])
            
            if cond_type == 'ctrl':
                conditions.append('ctrl')
            elif cond_type == 'single':
                g1 = np.random.randint(0, self.top_genes)
                conditions.append(f"gene_{g1}")
                # Use decoupled drift
                effect = orthogonal_biological_drift[g1, :] * np.random.normal(1.0, 0.2, self.top_genes)
                y_target[i, :] += effect
            elif cond_type == 'double':
                g1, g2 = np.random.choice(self.top_genes, 2, replace=False)
                conditions.append(f"gene_{g1}+gene_{g2}")
                effect1 = orthogonal_biological_drift[g1, :] * np.random.normal(1.0, 0.2, self.top_genes)
                effect2 = orthogonal_biological_drift[g2, :] * np.random.normal(1.0, 0.2, self.top_genes)
                
                # Decoupled epistatic synergy (mathematically invisible to the graph D)
                synergy = (orthogonal_biological_drift[g1, :] * orthogonal_biological_drift[g2, :]) * np.random.normal(2.0, 0.5, self.top_genes)
                y_target[i, :] += effect1 + effect2 + synergy
                
        self.X = torch.tensor(X_perturbed, dtype=torch.float32)
        self.y = torch.tensor(y_target, dtype=torch.float32)
        self.conditions = conditions
        print("Empirical dataset mapping complete (Circular leakage eliminated).")

    def create_splits(self) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
        n_cells = len(self.X)
        indices = np.arange(n_cells)
        np.random.shuffle(indices)
        
        split1 = int(0.6 * n_cells)
        split2 = int(0.75 * n_cells)
        split3 = int(0.9 * n_cells)
        
        train_idx = indices[:split1]
        seen2_idx = indices[split1:split2]
        seen1_idx = indices[split2:split3]
        seen0_idx = indices[split3:]
        
        def get_loader(idx, shuffle=False):
            ds = PerturbDataset(self.X[idx], self.y[idx], [self.conditions[i] for i in idx])
            return DataLoader(ds, batch_size=32, shuffle=shuffle)
            
        return get_loader(train_idx, True), get_loader(seen2_idx), get_loader(seen1_idx), get_loader(seen0_idx)
