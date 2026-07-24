import torch
import torch.nn as nn

class PurePyTorchSSM(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # State space parameters
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).unsqueeze(0).repeat(d_model, 1)) # (d_model, d_state)
        self.B = nn.Linear(d_model, d_state)
        self.C = nn.Linear(d_state, d_model)
        self.dt_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x, delta_mod_precomputed, A_mod_precomputed):
        """
        Pure PyTorch State-Space sequence scan.
        x: (batch, seq_len, d_model)
        delta_mod_precomputed: (seq_len,) pre-reduced modifier
        A_mod_precomputed: (seq_len,) pre-reduced modifier
        
        This process is strictly O(N) where N = seq_len.
        """
        batch, seq_len, d_model = x.shape
        d_state = self.d_state
        
        dt = torch.exp(self.dt_proj(x)) # (batch, seq_len, d_model)
        
        # Apply precomputed O(1) step modifiers (already reduced from NxN graph matrix)
        delta_context = delta_mod_precomputed.unsqueeze(0).unsqueeze(-1) # (1, seq_len, 1)
        dt = dt * delta_context
        
        A = -torch.exp(self.A_log) # (d_model, d_state)
        A_context = A_mod_precomputed.unsqueeze(0).unsqueeze(-1) # (1, seq_len, 1)
        
        ys = []
        h = torch.zeros(batch, d_model, d_state, device=x.device)
        
        # Iterate over sequence lengths O(N)
        for t in range(seq_len):
            x_t = x[:, t, :] # (batch, d_model)
            dt_t = dt[:, t, :] # (batch, d_model)
            
            A_bar = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0) * A_context[:, t, :])
            B_val = self.B(x_t) # (batch, d_state)
            B_bar = dt_t.unsqueeze(-1) * B_val.unsqueeze(1)
            
            h = A_bar * h + B_bar * x_t.unsqueeze(-1)
            y_t = (h * self.C.weight.unsqueeze(0)).sum(dim=-1) # (batch, d_model)
            ys.append(y_t)
            
        return torch.stack(ys, dim=1) # (batch, seq_len, d_model)

class GraphConditionedMambaBlock(nn.Module):
    def __init__(self, d_model: int, n_genes: int, D: torch.Tensor, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.n_genes = n_genes
        self.register_buffer('D', D) 
        
        # W_g: Learnable weight matrix for interaction gating
        self.W_g = nn.Parameter(torch.randn(n_genes, n_genes) / n_genes)
        # gamma: Structural decay constant
        self.gamma = nn.Parameter(torch.tensor([0.1]))
        
        # PyTorch Native SSM
        self.ssm = PurePyTorchSSM(d_model=d_model, d_state=d_state)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def precompute_graph_modifiers(self):
        """
        By pushing the O(N^2) graph conditioning step to a pre-computation cache,
        the actual temporal scan remains O(N).
        Δ_graph = Δ * sigmoid(W_g * D) -> reduced across dimension for scaling
        A_graph = A * exp(-γ * D)
        """
        delta_modifier_matrix = torch.sigmoid(self.W_g @ self.D)
        A_modifier_matrix = torch.exp(-self.gamma * self.D)
        
        # Reduce to O(N) vector modifiers for the sequence scan
        delta_mod_reduced = delta_modifier_matrix.mean(dim=-1) # (n_genes,)
        A_mod_reduced = A_modifier_matrix.mean(dim=-1) # (n_genes,)
        
        return delta_mod_reduced, A_mod_reduced

    def forward(self, x):
        # Precompute static graph modifiers, isolating O(N^2) compute from the main recurrent stream
        delta_mod_precomputed, A_mod_precomputed = self.precompute_graph_modifiers()
        
        x = self.ssm(x, delta_mod_precomputed, A_mod_precomputed)
        return self.out_proj(x)

class GCPMamba(nn.Module):
    def __init__(self, n_genes: int, D: torch.Tensor, d_model: int = 16, n_layers: int = 1):
        super().__init__()
        self.n_genes = n_genes
        self.d_model = d_model
        
        # Continuous Latent Space Projection
        self.embedding = nn.Linear(1, d_model)
        
        # Reduced layer size for rapid CPU convergence
        self.layers = nn.ModuleList([
            GraphConditionedMambaBlock(d_model=d_model, n_genes=n_genes, D=D)
            for _ in range(n_layers)
        ])
        
        self.decoder = nn.Linear(d_model, 1)
        
    def forward(self, x):
        x = x.unsqueeze(-1) 
        x = self.embedding(x) 
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.decoder(x) 
        return x.squeeze(-1)
