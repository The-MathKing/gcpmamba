import torch
import torch.nn as nn

class PurePyTorchSSM(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # State space parameters
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).unsqueeze(0).repeat(d_model, 1))
        self.B = nn.Linear(d_model, d_state)
        self.C = nn.Linear(d_state, d_model)
        self.dt_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x, delta_mod_precomputed=None, A_mod_precomputed=None):
        batch, seq_len, d_model = x.shape
        d_state = self.d_state
        
        dt = torch.exp(self.dt_proj(x))
        
        if delta_mod_precomputed is not None:
            delta_context = delta_mod_precomputed.unsqueeze(0).unsqueeze(-1)
            dt = dt * delta_context
        
        A = -torch.exp(self.A_log)
        if A_mod_precomputed is not None:
            A_context = A_mod_precomputed.unsqueeze(0).unsqueeze(-1)
        else:
            A_context = torch.ones(1, seq_len, 1, device=x.device)
        
        ys = []
        h = torch.zeros(batch, d_model, d_state, device=x.device)
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            dt_t = dt[:, t, :]
            
            A_bar = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0) * A_context[:, t, :])
            B_val = self.B(x_t)
            B_bar = dt_t.unsqueeze(-1) * B_val.unsqueeze(1)
            
            h = A_bar * h + B_bar * x_t.unsqueeze(-1)
            y_t = (h * self.C.weight.unsqueeze(0)).sum(dim=-1)
            ys.append(y_t)
            
        return torch.stack(ys, dim=1)

class MambaBlock(nn.Module):
    """
    Standard BaseMamba block (Ablation Control)
    """
    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = PurePyTorchSSM(d_model=d_model, d_state=d_state)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        res = x
        x = self.norm(x)
        x = self.ssm(x)
        x = self.out_proj(x)
        return x + res

class GraphConditionedMambaBlock(nn.Module):
    """
    GCP-Mamba block with topological conditioning
    """
    def __init__(self, d_model: int, n_genes: int, D: torch.Tensor, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.n_genes = n_genes
        self.register_buffer('D', D) 
        
        self.W_g = nn.Parameter(torch.randn(n_genes, n_genes) / np.sqrt(n_genes))
        self.gamma = nn.Parameter(torch.tensor([0.1]))
        
        self.norm = nn.LayerNorm(d_model)
        self.ssm = PurePyTorchSSM(d_model=d_model, d_state=d_state)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def precompute_graph_modifiers(self):
        delta_modifier_matrix = torch.sigmoid(self.W_g @ self.D)
        A_modifier_matrix = torch.exp(-self.gamma * self.D)
        
        delta_mod_reduced = delta_modifier_matrix.mean(dim=-1)
        A_mod_reduced = A_modifier_matrix.mean(dim=-1)
        
        return delta_mod_reduced, A_mod_reduced

    def forward(self, x):
        res = x
        x = self.norm(x)
        delta_mod_precomputed, A_mod_precomputed = self.precompute_graph_modifiers()
        
        x = self.ssm(x, delta_mod_precomputed, A_mod_precomputed)
        x = self.out_proj(x)
        return x + res

class BaseMamba(nn.Module):
    """
    Unconditioned model for Ablation Testing
    """
    def __init__(self, n_genes: int, d_model: int = 16, n_layers: int = 1):
        super().__init__()
        self.embedding = nn.Linear(1, d_model)
        self.layers = nn.ModuleList([MambaBlock(d_model=d_model) for _ in range(n_layers)])
        self.norm_f = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, 1)
        # Initialize decoder heavily to prevent zero-collapse
        nn.init.xavier_uniform_(self.decoder.weight, gain=1.0)
        nn.init.zeros_(self.decoder.bias)
        
    def forward(self, x):
        x = x.unsqueeze(-1) 
        x = self.embedding(x) 
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        x = self.decoder(x) 
        return x.squeeze(-1)

import numpy as np
class GCPMamba(nn.Module):
    """
    Graph-Conditioned target architecture
    """
    def __init__(self, n_genes: int, D: torch.Tensor, d_model: int = 16, n_layers: int = 1):
        super().__init__()
        self.embedding = nn.Linear(1, d_model)
        self.layers = nn.ModuleList([
            GraphConditionedMambaBlock(d_model=d_model, n_genes=n_genes, D=D)
            for _ in range(n_layers)
        ])
        self.norm_f = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, 1)
        # Initialize decoder heavily to prevent zero-collapse
        nn.init.xavier_uniform_(self.decoder.weight, gain=1.0)
        nn.init.zeros_(self.decoder.bias)
        
    def forward(self, x):
        x = x.unsqueeze(-1) 
        x = self.embedding(x) 
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        x = self.decoder(x) 
        return x.squeeze(-1)
