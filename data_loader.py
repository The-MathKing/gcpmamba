import numpy as np
import networkx as nx
import torch
from torch.utils.data import Dataset, DataLoader
import scanpy as sc
import json
import os
from scipy.sparse.csgraph import shortest_path

class PseudobulkDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor, conditions: list):
        """
        X: (N_conditions, d_model) — input perturbation indicators.
        y: (N_conditions, d_model) — target pseudobulk shifts.
        conditions: list of str
        """
        self.X = X
        self.y = y
        self.conditions = conditions
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.conditions[idx]

class DataEngine:
    def __init__(self, top_genes: int = 500, h5ad_path: str = 'data/norman/perturb_processed.h5ad', splits_path: str = 'splits_manifest.json'):
        self.top_genes = top_genes
        self.h5ad_path = h5ad_path
        self.splits_path = splits_path
        self.D = None
        self.X = None
        self.y = None
        self.conditions = None
        self.gene_names = None
        
        self.train_loader = None
        self.seen2_loader = None
        self.seen1_loader = None
        self.seen0_loader = None
        self.test_singles_loader = None
        
    def prepare_data(self):
        print(f"Loading {self.h5ad_path}...")
        adata = sc.read_h5ad(self.h5ad_path)
        
        # Load splits manifest
        with open(self.splits_path, 'r') as f:
            manifest = json.load(f)
            
        train_singles = manifest['train_singles']
        test_singles = manifest['test_singles']
        seen2_doubles = manifest['seen2_doubles']
        seen1_doubles = manifest['seen1_doubles']
        seen0_doubles = manifest['seen0_doubles']
        
        # Identify HVGs ONLY using training data (ctrl + train_singles)
        # This prevents data leakage into the feature selection
        train_mask = adata.obs['condition'].isin(['ctrl'] + train_singles)
        adata_train = adata[train_mask].copy()
        
        print("Identifying HVGs...")
        sc.pp.filter_genes(adata_train, min_cells=3)
        sc.pp.normalize_total(adata_train, target_sum=1e4)
        sc.pp.log1p(adata_train)
        sc.pp.highly_variable_genes(adata_train, n_top_genes=self.top_genes, subset=True)
        print("HVGs identified.")
        
        self.gene_names = adata_train.var_names.tolist()
        
        # Now apply the HVG filter and normalization to the FULL dataset
        print("Subsetting to HVGs...")
        adata = adata[:, self.gene_names].copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        
        print("Converting to dense array...")
        X_base = adata.X.toarray() if hasattr(adata.X, "toarray") else np.array(adata.X)
        
        print("Standardizing...")
        # Z-score standardize based on training data statistics
        X_train_base = X_base[train_mask]
        X_mean = X_train_base.mean(axis=0, keepdims=True)
        X_std = X_train_base.std(axis=0, keepdims=True) + 1e-8
        
        X_base_z = (X_base - X_mean) / X_std
        adata.X = X_base_z
        
        print("Computing control means...")
        # Compute control mean for pseudobulking
        ctrl_mask = adata.obs['condition'] == 'ctrl'
        ctrl_mean = X_base_z[ctrl_mask].mean(axis=0)
        
        # Build dense, continuous topological graph D based on empirical covariance
        # This replaces the unweighted shortest-path topology with a smooth continuous prior
        print("Computing continuous topological graph D...")
        cov_matrix = np.corrcoef(X_train_base.T)
        cov_matrix = np.nan_to_num(cov_matrix)
        D = 1.0 - np.abs(cov_matrix)
        self.D = torch.tensor(D, dtype=torch.float32)
        print("Continuous structural graph (1 - |corr|) extracted from empirical covariance.")
        
        print("Calculating condition-level pseudobulks...")
        # Calculate condition-level pseudobulks for ALL conditions
        unique_conditions = adata.obs['condition'].unique()
        
        # Create a mapping from gene name to index
        gene_to_idx = {g: i for i, g in enumerate(self.gene_names)}
        
        cond_X_list = []
        cond_y_list = []
        cond_names = []
        
        for cond in unique_conditions:
            if cond == 'ctrl': continue
            
            mask = adata.obs['condition'] == cond
            cond_mean = X_base_z[mask].mean(axis=0)
            delta_y = cond_mean - ctrl_mean
            
            # Construct input vector (perturbation indicator)
            # If a gene was perturbed and is in our HVG set, we set its input to 1.
            x_in = np.zeros(self.top_genes, dtype=np.float32)
            genes_perturbed = []
            if '+' in cond:
                genes_perturbed = cond.split('+')
            else:
                genes_perturbed = [cond.split('+')[0]] # single has +ctrl
                
            for g in genes_perturbed:
                if g in gene_to_idx:
                    x_in[gene_to_idx[g]] = 1.0
                    
            cond_X_list.append(x_in)
            cond_y_list.append(delta_y)
            cond_names.append(cond)
            
        X_tensor = torch.tensor(np.array(cond_X_list))
        y_tensor = torch.tensor(np.array(cond_y_list))
        
        def create_loader(cond_list, shuffle=False):
            idx = [i for i, c in enumerate(cond_names) if c in cond_list]
            if len(idx) == 0:
                # Return empty loader for empty splits
                return DataLoader(PseudobulkDataset(torch.empty(0, self.top_genes), torch.empty(0, self.top_genes), []), batch_size=64)
            ds = PseudobulkDataset(X_tensor[idx], y_tensor[idx], [cond_names[i] for i in idx])
            return DataLoader(ds, batch_size=64, shuffle=shuffle)
            
        self.train_loader = create_loader(train_singles + seen2_doubles, shuffle=True)
        self.seen2_loader = create_loader(seen2_doubles)
        self.seen1_loader = create_loader(seen1_doubles)
        self.seen0_loader = create_loader(seen0_doubles)
        self.test_singles_loader = create_loader(test_singles)
        
        print(f"Data Engine Ready!")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Seen 2/2 batches: {len(self.seen2_loader)}")
        print(f"Seen 1/2 batches: {len(self.seen1_loader)}")
        print(f"Seen 0/2 batches: {len(self.seen0_loader)}")
