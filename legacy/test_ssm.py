import torch
from model import PurePyTorchSSM

def test_manual_ssm_recurrence():
    """
    Manually computes the ZOH recurrence and compares against PurePyTorchSSM.
    This proves that the module executes a valid recurrent selective scan,
    addressing the reviewer's concern about a 'degenerate token-wise projection'.
    """
    B, L, D = 1, 3, 2  # Batch 1, Sequence 3, Dim 2
    N = 4  # State size
    
    # Force reproducible random weights
    torch.manual_seed(42)
    model = PurePyTorchSSM(d_model=D, d_state=N)
    model.eval()
    
    # Input sequence
    x = torch.randn(B, L, D)
    
    # Run forward pass
    with torch.no_grad():
        out = model(x)
        
    # Manual Recurrence
    # 1. Compute dt
    with torch.no_grad():
        dt = torch.sigmoid(model.dt_proj(x))  # (B, L, D)
        A = -torch.exp(model.A_log)  # (D, N)
        
        ys = []
        h = torch.zeros(B, D, N)
        for t in range(L):
            x_t = x[:, t, :]
            dt_t = dt[:, t, :]
            
            # A_bar: (B, D, N)
            A_bar = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))
            
            # Bx
            Bx = model.B(x_t)  # (B, N)
            B_bar = dt_t.unsqueeze(-1) * Bx.unsqueeze(1).expand(-1, D, -1)
            
            # h_t = A_bar * h_{t-1} + B_bar
            h = A_bar * h + B_bar
            
            # y_t = sum_n(C * h)
            y_t = (h * model.C.unsqueeze(0)).sum(-1)
            ys.append(y_t)
            
        manual_out = torch.stack(ys, dim=1)
        
    # Assert equivalence
    diff = torch.abs(out - manual_out).max().item()
    assert diff < 1e-5, f"Recurrence mismatch! Max diff: {diff}"
    print(f"SUCCESS: PurePyTorchSSM exactly matches manual recurrent scan. (Max diff: {diff:.2e})")

if __name__ == "__main__":
    test_manual_ssm_recurrence()
