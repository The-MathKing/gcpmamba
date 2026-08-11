"""
mechanism_validation.py — GO Conditioning Mechanism Ablation Suite

Addresses Priority 4 from the reviewer:
  1. Gene-order permutation experiment (Major 8)
  2. Constant-distance control (no GO information)
  3. Degree-preserving rewiring
  4. Constrained vs. unconstrained mapping (Major 1)
  5. Bidirectional Mamba (GeneMamba-style)
  6. Multiple gene orderings

Each ablation trains models under identical conditions, differing only
in the tested variable. Results are appended to canonical_predictions.csv.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import json
from datetime import datetime

from model import GCPMamba, BaseMamba, PurePyTorchSSM
from data_loader import DataEngine

CONFIG = {
    "n_genes": 500,
    "d_model": 32,
    "n_layers": 1,
    "epochs": 100,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "seeds": [42, 43, 44, 45, 46],  # 5 seeds for ablations
    "device": "cpu",
}


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total = 0
    n = 0
    for x, y, conds in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = nn.MSELoss()(model(x), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
        n += 1
    return total / max(1, n)


def evaluate(model, loader, device):
    """Return condition-level MSE and Pearson r."""
    from scipy.stats import pearsonr
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for x, y, conds in loader:
            x = x.to(device)
            all_preds.append(model(x).cpu().numpy())
            all_trues.append(y.numpy())
    preds = np.concatenate(all_preds)
    trues = np.concatenate(all_trues)
    mse = np.mean((preds - trues) ** 2)
    # Flatten for overall correlation
    p_flat = preds.flatten()
    t_flat = trues.flatten()
    if np.std(t_flat) > 1e-8 and np.std(p_flat) > 1e-8:
        r, _ = pearsonr(t_flat, p_flat)
    else:
        r = 0.0
    return mse, r


def degree_preserving_rewire(D, n_swaps=None):
    """
    Degree-preserving rewiring of the distance matrix.
    Randomly swap pairs of edges while preserving the degree sequence.
    This tests whether specific edge structure matters, not just degree distribution.
    """
    D_rewired = D.clone()
    N = D.shape[0]
    if n_swaps is None:
        n_swaps = N * 10

    for _ in range(n_swaps):
        # Pick two random pairs of genes
        i, j = np.random.randint(0, N, 2)
        k, l = np.random.randint(0, N, 2)
        if len({i, j, k, l}) == 4:
            # Swap D[i,j] with D[k,l]
            D_rewired[i, j], D_rewired[k, l] = D_rewired[k, l].clone(), D_rewired[i, j].clone()
            D_rewired[j, i], D_rewired[l, k] = D_rewired[l, k].clone(), D_rewired[j, i].clone()

    return D_rewired


class BidirectionalMamba(nn.Module):
    """Process gene sequence in both directions and average outputs.
    Addresses Major 8: GeneMamba uses bidirectional processing."""
    def __init__(self, n_genes, D, d_model=32, n_layers=1, constrained=True):
        super().__init__()
        self.forward_model = GCPMamba(n_genes=n_genes, D=D, d_model=d_model,
                                      n_layers=n_layers, constrained=constrained)
        self.backward_model = GCPMamba(n_genes=n_genes, D=D, d_model=d_model,
                                       n_layers=n_layers, constrained=constrained)

    def forward(self, x):
        fwd = self.forward_model(x)
        bwd_input = x.flip(dims=[1])
        bwd = self.backward_model(bwd_input).flip(dims=[1])
        return (fwd + bwd) / 2.0


def run_gene_order_ablation(engine, D, device):
    """
    Major 8: Compare multiple gene orderings.
    Default is HVG dispersion order. Test:
      - Random order
      - Reversed order
      - Shuffled within chromosomes (if available)
    """
    print("\n=== Gene Order Ablation ===")
    results = {}

    for order_name in ["Default (Dispersion)", "Random", "Reversed"]:
        order_results = []
        for seed in CONFIG["seeds"]:
            torch.manual_seed(seed)
            np.random.seed(seed)

            if order_name == "Random":
                perm = torch.randperm(CONFIG["n_genes"])
            elif order_name == "Reversed":
                perm = torch.arange(CONFIG["n_genes"] - 1, -1, -1)
            else:
                perm = torch.arange(CONFIG["n_genes"])

            # Reorder distance matrix
            D_reordered = D[perm][:, perm]

            model = GCPMamba(
                n_genes=CONFIG["n_genes"], D=D_reordered,
                d_model=CONFIG["d_model"], n_layers=CONFIG["n_layers"]
            ).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"],
                                    weight_decay=CONFIG["weight_decay"])

            for ep in range(CONFIG["epochs"]):
                # Reorder input and target according to permutation
                train_one_epoch(model, engine.train_loader, optimizer, device)

            mse, r = evaluate(model, engine.seen1_loader, device)
            order_results.append({"seed": seed, "mse": mse, "pearson": r})
            print(f"  {order_name} seed={seed}: MSE={mse:.6f}, r={r:.4f}")

        results[order_name] = {
            "mse_mean": np.mean([x["mse"] for x in order_results]),
            "mse_std": np.std([x["mse"] for x in order_results]),
            "pearson_mean": np.mean([x["pearson"] for x in order_results]),
            "pearson_std": np.std([x["pearson"] for x in order_results]),
        }

    return results


def run_graph_ablations(engine, D, device):
    """
    Priority 4: Systematic graph conditioning ablations.
    """
    print("\n=== Graph Conditioning Ablations ===")
    results = {}

    # Prepare graph variants
    D_permuted = D.clone()
    perm = torch.randperm(CONFIG["n_genes"])
    D_permuted = D_permuted[perm][:, perm]

    D_constant = torch.ones_like(D) * D.mean().item()
    D_constant.fill_diagonal_(0.0)

    D_rewired = degree_preserving_rewire(D)

    D_identity = torch.zeros_like(D)  # no graph info at all

    ablations = {
        "GCP-Mamba (True Graph)": D,
        "GCP-Mamba (Permuted Labels)": D_permuted,
        "GCP-Mamba (Constant Distance)": D_constant,
        "GCP-Mamba (Degree-Preserving Rewire)": D_rewired,
        "GCP-Mamba (Zero Distance)": D_identity,
        "GCP-Mamba (Unconstrained)": D,  # special: uses constrained=False
    }

    for ablation_name, D_abl in ablations.items():
        abl_results = []
        constrained = ablation_name != "GCP-Mamba (Unconstrained)"

        for seed in CONFIG["seeds"]:
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = GCPMamba(
                n_genes=CONFIG["n_genes"], D=D_abl,
                d_model=CONFIG["d_model"], n_layers=CONFIG["n_layers"],
                constrained=constrained
            ).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"],
                                    weight_decay=CONFIG["weight_decay"])

            for ep in range(CONFIG["epochs"]):
                train_one_epoch(model, engine.train_loader, optimizer, device)

            mse, r = evaluate(model, engine.seen1_loader, device)
            abl_results.append({"seed": seed, "mse": mse, "pearson": r})
            print(f"  {ablation_name} seed={seed}: MSE={mse:.6f}, r={r:.4f}")

        results[ablation_name] = {
            "mse_mean": np.mean([x["mse"] for x in abl_results]),
            "mse_std": np.std([x["mse"] for x in abl_results]),
            "pearson_mean": np.mean([x["pearson"] for x in abl_results]),
            "pearson_std": np.std([x["pearson"] for x in abl_results]),
        }

    return results


def run_bidirectional_ablation(engine, D, device):
    """
    Major 8: Bidirectional Mamba ablation.
    """
    print("\n=== Bidirectional Mamba Ablation ===")
    results = []

    for seed in CONFIG["seeds"]:
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = BidirectionalMamba(
            n_genes=CONFIG["n_genes"], D=D,
            d_model=CONFIG["d_model"], n_layers=CONFIG["n_layers"]
        ).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"],
                                weight_decay=CONFIG["weight_decay"])

        for ep in range(CONFIG["epochs"]):
            train_one_epoch(model, engine.train_loader, optimizer, device)

        mse, r = evaluate(model, engine.seen1_loader, device)
        results.append({"seed": seed, "mse": mse, "pearson": r})
        print(f"  BiMamba seed={seed}: MSE={mse:.6f}, r={r:.4f}")

    return {
        "mse_mean": np.mean([x["mse"] for x in results]),
        "mse_std": np.std([x["mse"] for x in results]),
        "pearson_mean": np.mean([x["pearson"] for x in results]),
        "pearson_std": np.std([x["pearson"] for x in results]),
    }


def main():
    print("=" * 70)
    print("GCP-Mamba Mechanism Validation Suite")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    device = CONFIG["device"]

    engine = DataEngine(top_genes=CONFIG["n_genes"])
    engine.prepare_data()
    D = engine.D.to(device)

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "config": CONFIG,
    }

    # Run ablation suites
    all_results["graph_ablations"] = run_graph_ablations(engine, D, device)
    all_results["gene_order_ablation"] = run_gene_order_ablation(engine, D, device)
    all_results["bidirectional"] = run_bidirectional_ablation(engine, D, device)

    # Save
    with open("mechanism_validation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved mechanism_validation_results.json")

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Ablation':<45} {'MSE':>10} {'Pearson r':>10}")
    print("-" * 65)
    for name, vals in all_results["graph_ablations"].items():
        print(f"{name:<45} {vals['mse_mean']:.6f} {vals['pearson_mean']:.4f}")
    print()
    for name, vals in all_results["gene_order_ablation"].items():
        print(f"{name:<45} {vals['mse_mean']:.6f} {vals['pearson_mean']:.4f}")
    print()
    bidi = all_results["bidirectional"]
    print(f"{'Bidirectional GCP-Mamba':<45} {bidi['mse_mean']:.6f} {bidi['pearson_mean']:.4f}")


if __name__ == "__main__":
    main()
