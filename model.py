import torch
import torch.nn as nn
import numpy as np


class PurePyTorchSSM(nn.Module):
    """
    Simplified, dimensionally-verified Selective SSM.

    State shape:  h ∈ (B, D, N_state)
    Input shape:  x ∈ (B, L, D)
    Output shape: y ∈ (B, L, D)

    For each of the D model dimensions, a 1-D SSM with N_state latent states
    is maintained.  The output is produced by contracting h over N_state with
    a learnable per-dim row vector c ∈ R^{N_state}.
    """
    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # A: strictly negative eigenvalues  (D, N)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
            .unsqueeze(0).repeat(d_model, 1)
        )
        # B projection: x_t -> N-dim context vector   (D, N)
        self.B = nn.Linear(d_model, d_state, bias=False)
        # C: output readout per model dim            (D, N)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        # Δ projection: x_t -> Δ_t ∈ (0,1)^D
        self.dt_proj = nn.Linear(d_model, d_model)
        nn.init.normal_(self.dt_proj.weight, std=0.01)
        nn.init.zeros_(self.dt_proj.bias)

    def forward(self, x, delta_mod=None, A_mod=None):
        """
        x          : (B, L, D)
        delta_mod  : (D,) optional multiplicative modifier on Δ
        A_mod      : (D,) optional multiplicative modifier on A eigenvalues
        returns    : (B, L, D)
        """
        B_sz, L, D = x.shape
        N = self.d_state

        # Δ_t ∈ (0,1)^D  — sigmoid keeps it bounded
        dt = torch.sigmoid(self.dt_proj(x))          # (B, L, D)
        if delta_mod is not None:
            # delta_mod: (D,)  →  broadcast over B and L
            dt = dt * delta_mod.unsqueeze(0).unsqueeze(0)

        A = -torch.exp(self.A_log)                   # (D, N), strictly negative
        if A_mod is not None:
            A = A * A_mod.unsqueeze(-1)              # (D, N) * (D, 1)

        ys = []
        h = torch.zeros(B_sz, D, N, device=x.device, dtype=x.dtype)

        for t in range(L):
            x_t  = x[:, t, :]                        # (B, D)
            dt_t = dt[:, t, :]                        # (B, D)

            # ZOH discretization
            # A_bar: (B, D, N)
            A_bar = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))
            # B_bar: Δ_t ⊙ (B x_t)  →  (B, D, N)
            Bx = self.B(x_t)                          # (B, N)
            B_bar = dt_t.unsqueeze(-1) * Bx.unsqueeze(1).expand(-1, D, -1)

            h = A_bar * h + B_bar                    # (B, D, N)

            # Output readout: y_t[b,d] = sum_n C[d,n] * h[b,d,n]
            # C: (D, N),  h: (B, D, N)
            y_t = (h * self.C.unsqueeze(0)).sum(-1)  # (B, D)
            ys.append(y_t)

        return torch.stack(ys, dim=1)                # (B, L, D)


class MambaBlock(nn.Module):
    """BaseMamba: pure sequence model, no graph conditioning (ablation control)."""
    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.norm     = nn.LayerNorm(d_model)
        self.ssm      = PurePyTorchSSM(d_model=d_model, d_state=d_state)
        self.out_proj = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):
        res = x
        x = self.ssm(self.norm(x))
        return self.out_proj(x) + res


class GraphConditionedMambaBlock(nn.Module):
    """GCP-Mamba block: topology-conditioned SSM with static M_Δ precomputation."""
    def __init__(self, d_model: int, n_genes: int, D: torch.Tensor, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.n_genes = n_genes
        self.register_buffer('D_mat', D)            # (N_genes, N_genes)

        self.W_g   = nn.Parameter(torch.randn(n_genes, n_genes) / np.sqrt(n_genes))
        self.gamma = nn.Parameter(torch.tensor([0.5]))

        # Project gene-space modifiers (N_genes,) → model-space (d_model,)
        self.delta_proj = nn.Linear(n_genes, d_model, bias=False)
        self.A_proj     = nn.Linear(n_genes, d_model, bias=False)
        nn.init.xavier_uniform_(self.delta_proj.weight)
        nn.init.xavier_uniform_(self.A_proj.weight)

        self.norm     = nn.LayerNorm(d_model)
        self.ssm      = PurePyTorchSSM(d_model=d_model, d_state=d_state)
        self.out_proj = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.out_proj.bias)

    def precompute_graph_modifiers(self):
        """
        Compute M_Δ ∈ R^{d_model} from the N×N graph prior.
        The O(N²) operations are FULLY ISOLATED here before the recurrent scan.
        """
        M_delta_gene = torch.sigmoid(self.W_g @ self.D_mat).mean(dim=-1)  # (N_genes,)
        A_mod_gene   = torch.exp(-self.gamma * self.D_mat.mean(dim=-1))   # (N_genes,)
        # Project to model-space and enforce absolute numeric stability
        M_delta = torch.sigmoid(self.delta_proj(M_delta_gene))   # (d_model,) bounded in (0,1)
        A_mod   = torch.exp(self.A_proj(A_mod_gene))             # (d_model,) strictly positive
        return M_delta, A_mod

    def forward(self, x):
        res = x
        x = self.norm(x)
        delta_mod, A_mod = self.precompute_graph_modifiers()
        x = self.ssm(x, delta_mod=delta_mod, A_mod=A_mod)
        return self.out_proj(x) + res


# ──────────────────────────────────────────────────
# Top-level model definitions
# ──────────────────────────────────────────────────

class BaseMamba(nn.Module):
    """Ablation control: identical capacity to GCPMamba but no graph conditioning."""
    def __init__(self, n_genes: int, d_model: int = 32, n_layers: int = 2):
        super().__init__()
        self.embedding = nn.Linear(1, d_model)
        self.layers    = nn.ModuleList([MambaBlock(d_model=d_model) for _ in range(n_layers)])
        self.norm_f    = nn.LayerNorm(d_model)
        self.decoder   = nn.Linear(d_model, 1)
        nn.init.xavier_normal_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, x):
        # x: (B, L)
        x = self.embedding(x.unsqueeze(-1))    # (B, L, d_model)
        for layer in self.layers:
            x = layer(x)
        return self.decoder(self.norm_f(x)).squeeze(-1)   # (B, L)


class GCPMamba(nn.Module):
    """Graph-Conditioned Perturbation Mamba — the primary architecture."""
    def __init__(self, n_genes: int, D: torch.Tensor, d_model: int = 32, n_layers: int = 2):
        super().__init__()
        self.embedding = nn.Linear(1, d_model)
        self.layers = nn.ModuleList([
            GraphConditionedMambaBlock(d_model=d_model, n_genes=n_genes, D=D)
            for _ in range(n_layers)
        ])
        self.norm_f  = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, 1)
        nn.init.xavier_normal_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, x):
        # x: (B, L)
        x = self.embedding(x.unsqueeze(-1))    # (B, L, d_model)
        for layer in self.layers:
            x = layer(x)
        return self.decoder(self.norm_f(x)).squeeze(-1)   # (B, L)
