"""GCP-Mamba v2: perturbation-conditioned selective state-space model over genes.

Each of the G measured genes is one token. A perturbation set P enters the
model three ways:
  1. an indicator on the token of every targeted gene (all targets are measured
     genes, so no perturbation is ever silently dropped);
  2. a set embedding sum_{g in P} f(e_g), where e_g is the control-cell
     expression embedding of gene g -- this is what lets the model make
     predictions for targets never perturbed during training;
  3. the graph signal c(P) = sum_k alpha^k  W^k 1_P  (diffusion of the
     perturbation over the control-cell co-expression graph W), which is added
     to the pre-activation of the step size Delta of every token:
         Delta_i = softplus(dt_proj(x_i) + w_c * c_i(P)).
     Tokens close to a perturbed gene therefore take larger steps (write more of
     their input into, and forget more of, the recurrent state).

The scan is bidirectional and genes are placed along the Fiedler ordering of W,
so co-expressed genes are neighbours in the sequence.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

CHUNK = 16         # chunk length of the parallel scan
MIN_LOG_A = -5.0   # clamp on per-step log decay (a >= 0.0067); CHUNK * |MIN_LOG_A| < 88 keeps exp() finite in fp32


def selective_scan(log_a, b):
    """h_t = exp(log_a_t) * h_{t-1} + b_t along dim 1, for (B, L, D, N) tensors.

    Exact (up to the clamp on log_a) chunked scan: a cumulative-sum solution
    inside each chunk and a short sequential recurrence across chunk states.
    Memory is O(B L D N); no L x L object is formed.
    """
    B, L, D, N = b.shape
    pad = (-L) % CHUNK
    if pad:
        log_a = F.pad(log_a, (0, 0, 0, 0, 0, pad))
        b = F.pad(b, (0, 0, 0, 0, 0, pad))
    C = (L + pad) // CHUNK
    log_a = log_a.clamp(min=MIN_LOG_A).view(B, C, CHUNK, D, N)
    b = b.view(B, C, CHUNK, D, N)
    cum = torch.cumsum(log_a, dim=2)
    h_local = torch.exp(cum) * torch.cumsum(torch.exp(-cum) * b, dim=2)
    chunk_decay = torch.exp(cum[:, :, -1])            # (B, C, D, N)
    chunk_end = h_local[:, :, -1]
    carry = [torch.zeros_like(chunk_end[:, 0])]
    for c in range(C - 1):
        carry.append(chunk_decay[:, c] * carry[-1] + chunk_end[:, c])
    carry = torch.stack(carry, dim=1)                  # state entering each chunk
    h = h_local + torch.exp(cum) * carry.unsqueeze(2)
    return h.reshape(B, C * CHUNK, D, N)[:, :L]


class SelectiveSSM(nn.Module):
    """Mamba-style selective SSM (diagonal A, input-dependent Delta, B, C, gated output)."""

    def __init__(self, d_model, d_state, graph_conditioned):
        super().__init__()
        self.in_proj = nn.Linear(d_model, 2 * d_model)
        self.dt_proj = nn.Linear(d_model, d_model)
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()).repeat(d_model, 1))
        self.D_skip = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(d_model, d_model)
        # w_c: how strongly graph proximity to the perturbation modulates Delta
        self.w_c = nn.Parameter(torch.zeros(d_model)) if graph_conditioned else None
        with torch.no_grad():  # Mamba initialisation: Delta in [1e-3, 1e-1]
            dt = torch.exp(torch.rand(d_model) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def forward(self, x, c=None):
        u, z = self.in_proj(x).chunk(2, dim=-1)
        u = F.silu(u)
        dt_pre = self.dt_proj(u)
        if self.w_c is not None:
            dt_pre = dt_pre + c.unsqueeze(-1) * self.w_c
        dt = F.softplus(dt_pre)                                   # (B, L, D)
        A = -torch.exp(self.A_log)                                # (D, N)
        log_a = dt.unsqueeze(-1) * A                              # ZOH: A_bar = exp(Delta A)
        b = (dt * u).unsqueeze(-1) * self.B_proj(u).unsqueeze(2)  # Euler B_bar = Delta B
        h = selective_scan(log_a, b)
        y = (h * self.C_proj(u).unsqueeze(2)).sum(-1) + self.D_skip * u
        return self.out_proj(y * F.silu(z))


class BiSSMBlock(nn.Module):
    def __init__(self, d_model, d_state, graph_conditioned):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fwd = SelectiveSSM(d_model, d_state, graph_conditioned)
        self.bwd = SelectiveSSM(d_model, d_state, graph_conditioned)

    def forward(self, x, c):
        h = self.norm(x)
        y = self.fwd(h, c) + self.bwd(h.flip(1), c.flip(1)).flip(1)
        return x + y


class TokenMLPBlock(nn.Module):
    """Ablation: same inputs, no cross-gene recurrence (each gene processed independently)."""

    def __init__(self, d_model, *_):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, d_model))

    def forward(self, x, c):
        return x + self.mlp(self.norm(x))


class GCPMamba(nn.Module):
    """Variants (see run_benchmark.VARIANTS):
    graph_in_delta -- c(P) modulates Delta (the proposed conditioning)
    graph_in_input -- c(P) is also added to the token features
    block          -- 'ssm' (bidirectional selective scan) or 'mlp' (no recurrence)
    """

    def __init__(self, n_genes, emb, ctrl_mean, d_model=32, d_state=8, n_layers=2,
                 graph_in_delta=True, graph_in_input=True, block='ssm'):
        super().__init__()
        self.register_buffer('emb', emb)                # (G, E) control-cell gene embeddings
        self.register_buffer('ctrl', ctrl_mean)          # (G,)
        self.graph_in_input = graph_in_input
        self.gene_emb = nn.Parameter(torch.randn(n_genes, d_model) * 0.02)
        self.gene_in = nn.Linear(emb.shape[1] + 1, d_model)
        self.pert_enc = nn.Sequential(nn.Linear(emb.shape[1], 2 * d_model), nn.GELU(),
                                      nn.Linear(2 * d_model, d_model))
        self.target_flag = nn.Parameter(torch.randn(d_model) * 0.02)
        self.graph_in = nn.Linear(1, d_model) if graph_in_input else None
        Block = BiSSMBlock if block == 'ssm' else TokenMLPBlock
        self.layers = nn.ModuleList([Block(d_model, d_state, graph_in_delta) for _ in range(n_layers)])
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)
        self.gene_bias = nn.Parameter(torch.zeros(n_genes))

    def forward(self, P, c):
        """P: (B, G) 0/1 perturbation indicator; c: (B, G) graph signal. Returns (B, G) predicted delta."""
        base = self.gene_emb + self.gene_in(torch.cat([self.emb, self.ctrl.unsqueeze(-1)], -1))
        pert = P @ self.pert_enc(self.emb)                       # (B, d): sum of target embeddings
        x = base.unsqueeze(0) + pert.unsqueeze(1) + P.unsqueeze(-1) * self.target_flag
        if self.graph_in is not None:
            x = x + self.graph_in(c.unsqueeze(-1))
        for layer in self.layers:
            x = layer(x, c)
        return self.head(self.norm_f(x)).squeeze(-1) + self.gene_bias
