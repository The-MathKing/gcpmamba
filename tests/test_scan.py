"""The chunked parallel scan must match the naive sequential recurrence."""
import torch
from gcpm.model import selective_scan


def naive(log_a, b):
    h = torch.zeros_like(b[:, 0])
    out = []
    for t in range(b.shape[1]):
        h = torch.exp(log_a[:, t]) * h + b[:, t]
        out.append(h)
    return torch.stack(out, 1)


def test_scan_matches_recurrence():
    torch.manual_seed(0)
    for L in (1, 7, 8, 33, 257):
        log_a = -torch.rand(2, L, 4, 3) * 4.9  # inside the MIN_LOG_A clamp
        b = torch.randn(2, L, 4, 3)
        assert torch.allclose(selective_scan(log_a, b), naive(log_a, b), atol=1e-5, rtol=1e-4)


if __name__ == '__main__':
    test_scan_matches_recurrence()
    print('ok')
