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
        
    def forward(self, x, delta_mod, A_mod):
        """
        Pure PyTorch State-Space sequence scan.
        x: (batch, seq_len, d_model)
        delta_mod: (n_genes, n_genes) modifier for timestep sizes
        A_mod: (n_genes, n_genes) structural decay for state transitions
        """
        batch, seq_len, d_model = x.shape
        d_state = self.d_state
        
        # We will flatten the state scan across the sequence dimension
        # x shape is (batch, seq_len, d_model)
        
        dt = torch.exp(self.dt_proj(x)) # (batch, seq_len, d_model)
        
        # Apply the graph modifiers to dt. delta_mod is structurally based on D
        # To simplify projection, we multiply dt across the seq dim
        delta_context = delta_mod.mean(dim=-1).unsqueeze(0).unsqueeze(-1) # (1, seq_len, 1)
        dt = dt * delta_context
        
        A = -torch.exp(self.A_log) # (d_model, d_state)
        # Apply A graph structural decay
        A_context = A_mod.mean(dim=-1).unsqueeze(0).unsqueeze(-1) # (1, seq_len, 1)
        
        ys = []
        h = torch.zeros(batch, d_model, d_state, device=x.device)
        
        # Iterate over sequence lengths
        for t in range(seq_len):
            x_t = x[:, t, :] # (batch, d_model)
            dt_t = dt[:, t, :] # (batch, d_model)
            
            # Discretize A and B
            # A_bar: (batch, d_model, d_state)
            A_bar = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0) * A_context[:, t, :])
            # B_bar: (batch, d_model, d_state)
            B_val = self.B(x_t) # (batch, d_state)
            # Expand to (batch, d_model, d_state)
            B_bar = dt_t.unsqueeze(-1) * B_val.unsqueeze(1)
            
            # Update state
            h = A_bar * h + B_bar * x_t.unsqueeze(-1)
            
            # Output projection
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
        
    def compute_graph_modifiers(self):
        """
        Calculates internal state-space modifiers.
        Δ_graph = Δ * sigmoid(W_g * D)
        A_graph = A * exp(-γ * D)
        """
        delta_modifier = torch.sigmoid(self.W_g @ self.D)
        A_modifier = torch.exp(-self.gamma * self.D)
        return delta_modifier, A_modifier

    def forward(self, x):
        delta_mod, A_mod = self.compute_graph_modifiers()
        
        x = self.ssm(x, delta_mod, A_mod)
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
