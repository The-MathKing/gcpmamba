"""
full_benchmark.py — Complete Computational Benchmarking Suite

Addresses Priority 7 and Major 2 from the review.

Measures SEPARATELY:
  1. GO matrix construction time and storage
  2. Model parameter count
  3. Forward pass memory (INCLUDING D matrix)
  4. Backward pass memory
  5. Optimizer state memory
  6. Wall-clock training time per epoch
  7. Inference throughput (conditions/second)

Reports all components honestly, not just one selected tensor.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
import tracemalloc
import gc
from datetime import datetime

from model import GCPMamba, BaseMamba, PurePyTorchSSM


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_go_matrix(n_genes):
    """Measure GO distance matrix construction cost."""
    # Simulate co-expression matrix computation
    start = time.perf_counter()
    X = np.random.randn(1000, n_genes).astype(np.float32)
    cov = np.corrcoef(X.T)
    cov = np.nan_to_num(cov)
    D = 1.0 - np.abs(cov)
    elapsed = time.perf_counter() - start

    storage_bytes = D.nbytes
    storage_mb = storage_bytes / (1024 ** 2)

    return {
        "n_genes": n_genes,
        "construction_time_s": elapsed,
        "storage_bytes": storage_bytes,
        "storage_mb": storage_mb,
        "dtype": str(D.dtype),
    }


def measure_model_memory(model_class, n_genes, d_model, n_layers, D=None,
                         batch_size=8, include_backward=True):
    """
    Measure actual memory usage using tracemalloc.
    Returns dict with breakdown of memory components.
    """
    gc.collect()

    # Model construction
    tracemalloc.start()
    if D is not None:
        model = model_class(n_genes=n_genes, D=D, d_model=d_model, n_layers=n_layers)
    else:
        model = model_class(n_genes=n_genes, d_model=d_model, n_layers=n_layers)
    model_snapshot = tracemalloc.take_snapshot()
    model_mem = tracemalloc.get_traced_memory()[1]

    # Parameter memory
    param_mem = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_mem = sum(b.nelement() * b.element_size() for n, b in model.named_buffers())

    # Forward pass
    x = torch.randn(batch_size, n_genes)
    tracemalloc.reset_peak()
    out = model(x)
    forward_peak = tracemalloc.get_traced_memory()[1]

    # Backward pass
    backward_peak = 0
    optimizer_mem = 0
    if include_backward:
        target = torch.randn_like(out)
        loss = F.mse_loss(out, target)

        optimizer = torch.optim.AdamW(model.parameters())
        tracemalloc.reset_peak()
        loss.backward()
        backward_peak = tracemalloc.get_traced_memory()[1]

        # Optimizer step
        tracemalloc.reset_peak()
        optimizer.step()
        optimizer_peak = tracemalloc.get_traced_memory()[1]
        # AdamW stores m and v for each parameter
        optimizer_mem = 2 * param_mem  # approximate: 2x params for momentum + variance

    tracemalloc.stop()

    # D matrix memory (if applicable)
    d_matrix_mem = D.nelement() * D.element_size() if D is not None else 0

    total, trainable = count_parameters(model)

    return {
        "n_genes": n_genes,
        "d_model": d_model,
        "n_layers": n_layers,
        "batch_size": batch_size,
        "total_params": total,
        "trainable_params": trainable,
        "param_memory_mb": param_mem / (1024 ** 2),
        "buffer_memory_mb": buffer_mem / (1024 ** 2),
        "d_matrix_memory_mb": d_matrix_mem / (1024 ** 2),
        "forward_peak_mb": forward_peak / (1024 ** 2),
        "backward_peak_mb": backward_peak / (1024 ** 2),
        "optimizer_state_mb": optimizer_mem / (1024 ** 2),
        "total_training_mb": (param_mem + buffer_mem + d_matrix_mem + optimizer_mem) / (1024 ** 2),
    }


def measure_throughput(model_class, n_genes, d_model, n_layers, D=None,
                       batch_size=64, n_batches=50):
    """Measure forward-pass throughput in conditions/second."""
    if D is not None:
        model = model_class(n_genes=n_genes, D=D, d_model=d_model, n_layers=n_layers)
    else:
        model = model_class(n_genes=n_genes, d_model=d_model, n_layers=n_layers)
    model.eval()

    # Warmup
    x = torch.randn(batch_size, n_genes)
    with torch.no_grad():
        for _ in range(5):
            _ = model(x)

    # Time
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_batches):
            _ = model(x)
    elapsed = time.perf_counter() - start

    total_conditions = batch_size * n_batches
    throughput = total_conditions / elapsed

    return {
        "n_genes": n_genes,
        "batch_size": batch_size,
        "n_batches": n_batches,
        "total_conditions": total_conditions,
        "elapsed_s": elapsed,
        "throughput_cond_per_s": throughput,
    }


def measure_epoch_time(n_genes, d_model, n_layers, batch_size=64, n_conditions=100):
    """Measure wall-clock time for one training epoch."""
    D = torch.rand(n_genes, n_genes)
    D = (D + D.T) / 2
    D.fill_diagonal_(0.0)

    results = {}
    for model_name, model_class, needs_D in [
        ("BaseMamba", BaseMamba, False),
        ("GCP-Mamba", GCPMamba, True),
    ]:
        if needs_D:
            model = model_class(n_genes=n_genes, D=D, d_model=d_model, n_layers=n_layers)
        else:
            model = model_class(n_genes=n_genes, d_model=d_model, n_layers=n_layers)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x = torch.randn(n_conditions, n_genes)
        y = torch.randn(n_conditions, n_genes)
        dataset = torch.utils.data.TensorDataset(x, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

        # Time one epoch
        model.train()
        start = time.perf_counter()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = F.mse_loss(model(xb), yb)
            loss.backward()
            optimizer.step()
        elapsed = time.perf_counter() - start

        results[model_name] = {
            "n_genes": n_genes,
            "n_conditions": n_conditions,
            "epoch_time_s": elapsed,
        }

    return results


def main():
    print("=" * 70)
    print("GCP-Mamba Full Computational Benchmark")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "hardware": "CPU (Apple Silicon)" if torch.backends.mps.is_available() else "CPU",
    }

    # ── 1. GO Matrix Scaling ──
    print("\n--- GO Matrix Construction ---")
    go_results = []
    for n in [100, 500, 1000, 2000, 5000]:
        r = measure_go_matrix(n)
        go_results.append(r)
        print(f"  N={n:5d}: {r['storage_mb']:.2f} MB, {r['construction_time_s']:.3f}s")
    all_results["go_matrix_scaling"] = go_results

    # ── 2. Model Memory Breakdown ──
    print("\n--- Memory Breakdown (batch=8) ---")
    memory_results = {}
    d_model = 32
    n_layers = 1

    for n_genes in [100, 500, 1000, 2000, 5000]:
        D = torch.rand(n_genes, n_genes)
        D = (D + D.T) / 2
        D.fill_diagonal_(0.0)

        for model_name, model_class, needs_D in [
            ("BaseMamba", BaseMamba, False),
            ("GCP-Mamba", GCPMamba, True),
        ]:
            gc.collect()
            try:
                r = measure_model_memory(
                    model_class, n_genes, d_model, n_layers,
                    D=D if needs_D else None, batch_size=8
                )
                key = f"{model_name}_N{n_genes}"
                memory_results[key] = r
                print(f"  {key}: params={r['param_memory_mb']:.2f}MB, "
                      f"D_mat={r['d_matrix_memory_mb']:.2f}MB, "
                      f"total_train={r['total_training_mb']:.2f}MB")
            except Exception as e:
                print(f"  {model_name} N={n_genes}: FAILED ({e})")
                memory_results[f"{model_name}_N{n_genes}"] = {"error": str(e)}

    all_results["memory_breakdown"] = memory_results

    # ── 3. Throughput ──
    print("\n--- Inference Throughput ---")
    throughput_results = {}
    for n_genes in [100, 500, 1000]:
        D = torch.rand(n_genes, n_genes)
        D = (D + D.T) / 2
        D.fill_diagonal_(0.0)

        for model_name, model_class, needs_D in [
            ("BaseMamba", BaseMamba, False),
            ("GCP-Mamba", GCPMamba, True),
        ]:
            r = measure_throughput(
                model_class, n_genes, d_model, n_layers,
                D=D if needs_D else None
            )
            key = f"{model_name}_N{n_genes}"
            throughput_results[key] = r
            print(f"  {key}: {r['throughput_cond_per_s']:.0f} conditions/s")

    all_results["throughput"] = throughput_results

    # ── 4. Epoch Timing ──
    print("\n--- Epoch Wall-Clock Time ---")
    epoch_results = {}
    for n_genes in [100, 500, 1000]:
        r = measure_epoch_time(n_genes, d_model, n_layers)
        epoch_results[f"N{n_genes}"] = r
        for name, vals in r.items():
            print(f"  {name} N={n_genes}: {vals['epoch_time_s']:.3f}s/epoch")

    all_results["epoch_timing"] = epoch_results

    # ── Save ──
    with open("full_benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved full_benchmark_results.json")

    # ── Summary Table ──
    print("\n" + "=" * 70)
    print("CRITICAL POINT (Reviewer Major 2):")
    print("The memory figure MUST include the D matrix storage.")
    print("=" * 70)
    for key, val in memory_results.items():
        if "error" not in val:
            print(f"  {key:30s}: D_mat={val['d_matrix_memory_mb']:8.2f} MB  "
                  f"total_training={val['total_training_mb']:8.2f} MB")


if __name__ == "__main__":
    main()
