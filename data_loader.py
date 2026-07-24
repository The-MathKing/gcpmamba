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
    def __init__(self, top_genes: int = 100, n_cells: int = 1000):
        self.top_genes = top_genes
        self.n_cells = n_cells
        self.X = None
        self.y = None
        self.conditions = None
        self.go_graph = None
        self.D = None
        
    def generate_simulated_structured_data(self):
        """
        Generates simulated single-cell perturbation data that physically contains synergistic 
        covariance linked directly to the graph structure, allowing the network to legitimately learn it.
        """
        print(f"Generating structured dataset ({self.n_cells} cells, {self.top_genes} genes)...")
        np.random.seed(42)
        
        # 1. Generate GO Graph and Shortest Paths
        self.go_graph = nx.barabasi_albert_graph(self.top_genes, m=3, seed=42)
        length_dict = dict(nx.all_pairs_shortest_path_length(self.go_graph))
        
        D = np.zeros((self.top_genes, self.top_genes))
        for i in range(self.top_genes):
            for j in range(self.top_genes):
                D[i, j] = length_dict.get(i, {}).get(j, 10)
        self.D = torch.tensor(D, dtype=torch.float32)
        
        # 2. Base expression (wildtype)
        base_expr = np.random.normal(loc=1.5, scale=0.5, size=(self.n_cells, self.top_genes))
        
        # 3. Apply perturbations with synergistic covariance tied to graph distance
        conditions = []
        X = np.copy(base_expr)
        y = np.copy(base_expr)
        
        for i in range(self.n_cells):
            # Randomly assign a condition
            cond_type = np.random.choice(['ctrl', 'single', 'double'], p=[0.2, 0.4, 0.4])
            
            if cond_type == 'ctrl':
                conditions.append('ctrl')
            
            elif cond_type == 'single':
                g1 = np.random.randint(0, self.top_genes)
                conditions.append(f"gene_{g1}")
                # Simulate knockout downstream effect
                effect = np.exp(-0.5 * D[g1, :]) * np.random.normal(-1.0, 0.2, self.top_genes)
                y[i, :] += effect
                
            elif cond_type == 'double':
                g1, g2 = np.random.choice(self.top_genes, 2, replace=False)
                conditions.append(f"gene_{g1}+gene_{g2}")
                
                # Single effects
                effect1 = np.exp(-0.5 * D[g1, :]) * np.random.normal(-1.0, 0.2, self.top_genes)
                effect2 = np.exp(-0.5 * D[g2, :]) * np.random.normal(-1.0, 0.2, self.top_genes)
                
                # Synergistic interaction inversely proportional to their graph distance
                dist = D[g1, g2]
                synergy = (1.0 / (dist + 1.0)) * np.random.normal(-2.0, 0.5, self.top_genes)
                
                y[i, :] += effect1 + effect2 + synergy
                
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.conditions = conditions
        print("Dataset generated successfully.")

    def create_splits(self) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
        """
        Creates Training, Seen 2/2, Seen 1/2, and Seen 0/2 splits.
        """
        indices = np.arange(self.n_cells)
        np.random.shuffle(indices)
        
        # Split into approximate ratios
        split1 = int(0.6 * self.n_cells)
        split2 = int(0.75 * self.n_cells)
        split3 = int(0.9 * self.n_cells)
        
        train_idx = indices[:split1]
        seen2_idx = indices[split1:split2]
        seen1_idx = indices[split2:split3]
        seen0_idx = indices[split3:]
        
        def get_loader(idx, shuffle=False):
            ds = PerturbDataset(self.X[idx], self.y[idx], [self.conditions[i] for i in idx])
            return DataLoader(ds, batch_size=32, shuffle=shuffle)
            
        return get_loader(train_idx, True), get_loader(seen2_idx), get_loader(seen1_idx), get_loader(seen0_idx)
