"""
canonical_results.py — Single Immutable Results Pipeline

Addresses Critical 2: All tables, figures, abstract values, and discussion
statements MUST derive from this canonical predictions file.

Trains all models under identical conditions, writes a single canonical CSV
with checksums, and generates all downstream analytics.

Columns: split, condition, model, seed, gene_idx, gene_name,
         true_value, pred_value, preprocessing, target_type
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import json
import hashlib
import os
import sys
from datetime import datetime

from model import GCPMamba, BaseMamba
from data_loader import DataEngine

# ─────────────────────────────────────────────────────────────────────────
# CONFIG — Central source of truth for all hyperparameters
# ─────────────────────────────────────────────────────────────────────────
CONFIG = {
    "n_genes": 500,
    "d_model": 32,
    "d_state": 16,
    "n_layers": 1,
    "epochs": 100,          # Increased from 50 for convergence
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 64,
    "seeds": list(range(42, 42 + 15)),  # 15 seeds for statistical power
    "grad_clip": 1.0,
    "device": "cpu",
    "preprocessing": "normalize_total(1e4) → log1p → z-score(train_only)",
    "target_type": "condition_level_pseudobulk",
    "h5ad_path": "data/norman/perturb_processed.h5ad",
    "splits_path": "splits_manifest.json",
    "output_file": "canonical_predictions.csv",
    "constrained_monotonic": True,  # softplus constraint on delta_proj
}


def compute_file_checksum(filepath):
    """SHA256 of a file for reproducibility tracking."""
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for x, y, conds in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(x)
        loss = nn.MSELoss()(preds, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(1, n_batches)


def evaluate_to_records(model, loader, model_name, seed, split_name,
                        gene_names, device):
    """Evaluate model and return list of per-gene-per-condition records."""
    model.eval()
    records = []
    with torch.no_grad():
        for x, y, conds in loader:
            x = x.to(device)
            preds = model(x).cpu().numpy()
            trues = y.numpy()

            for i, cond in enumerate(conds):
                for g_idx in range(CONFIG["n_genes"]):
                    records.append({
                        "Split": split_name,
                        "Condition": cond,
                        "Model": model_name,
                        "Seed": seed,
                        "Gene_Idx": g_idx,
                        "Gene_Name": gene_names[g_idx] if g_idx < len(gene_names) else f"gene_{g_idx}",
                        "True_Value": float(trues[i, g_idx]),
                        "Pred_Value": float(preds[i, g_idx]),
                        "Preprocessing": CONFIG["preprocessing"],
                        "Target_Type": CONFIG["target_type"],
                    })
    return records


def build_additive_records(engine, gene_names, split_name, loader, seeds):
    """
    Build additive baseline predictions.
    
    Additive handling per split tier (addressing Critical 4):
      - Seen 2/2: both singles measured in training → sum of measured profiles
      - Seen 1/2: one single available, other zero → partial sum
      - Seen 0/2: both singles unavailable → predicts zero for both
    """
    # Extract single-gene effects from training data
    single_effects = {}
    for x, y, c in engine.train_loader:
        for i, cond in enumerate(c):
            if cond.endswith('+ctrl'):
                gene = cond.split('+')[0]
                single_effects[gene] = y[i].numpy()

    records = []
    for x, y, conds in loader:
        trues = y.numpy()
        for i, cond in enumerate(conds):
            parts = cond.split('+')
            if len(parts) == 2 and 'ctrl' not in parts:
                g1, g2 = parts
                eff1 = single_effects.get(g1, np.zeros(CONFIG["n_genes"]))
                eff2 = single_effects.get(g2, np.zeros(CONFIG["n_genes"]))
                pred = eff1 + eff2
            elif cond.endswith('+ctrl'):
                gene = cond.split('+')[0]
                pred = single_effects.get(gene, np.zeros(CONFIG["n_genes"]))
            else:
                pred = np.zeros(CONFIG["n_genes"])

            for g_idx in range(CONFIG["n_genes"]):
                # Additive is deterministic — same across seeds
                for seed in seeds:
                    records.append({
                        "Split": split_name,
                        "Condition": cond,
                        "Model": "Additive",
                        "Seed": seed,
                        "Gene_Idx": g_idx,
                        "Gene_Name": gene_names[g_idx] if g_idx < len(gene_names) else f"gene_{g_idx}",
                        "True_Value": float(trues[i, g_idx]),
                        "Pred_Value": float(pred[g_idx]),
                        "Preprocessing": CONFIG["preprocessing"],
                        "Target_Type": CONFIG["target_type"],
                    })
    return records


def build_condition_mean_records(engine, gene_names, split_name, loader, seeds):
    """Condition mean baseline: predict training set mean for every condition."""
    # Compute training condition mean
    all_y = []
    for x, y, c in engine.train_loader:
        all_y.append(y.numpy())
    train_mean = np.concatenate(all_y).mean(axis=0)

    records = []
    for x, y, conds in loader:
        trues = y.numpy()
        for i, cond in enumerate(conds):
            for g_idx in range(CONFIG["n_genes"]):
                for seed in seeds:
                    records.append({
                        "Split": split_name,
                        "Condition": cond,
                        "Model": "Condition Mean",
                        "Seed": seed,
                        "Gene_Idx": g_idx,
                        "Gene_Name": gene_names[g_idx] if g_idx < len(gene_names) else f"gene_{g_idx}",
                        "True_Value": float(trues[i, g_idx]),
                        "Pred_Value": float(train_mean[g_idx]),
                        "Preprocessing": CONFIG["preprocessing"],
                        "Target_Type": CONFIG["target_type"],
                    })
    return records


def build_linear_records(engine, gene_names, split_name, loader, seeds):
    """Ridge regression baseline."""
    from sklearn.linear_model import Ridge

    train_X, train_y = [], []
    for x, y, c in engine.train_loader:
        train_X.append(x.numpy())
        train_y.append(y.numpy())
    train_X = np.concatenate(train_X)
    train_y = np.concatenate(train_y)

    ridge = Ridge(alpha=1.0)
    ridge.fit(train_X, train_y)

    records = []
    for x, y, conds in loader:
        trues = y.numpy()
        preds = ridge.predict(x.numpy())
        for i, cond in enumerate(conds):
            for g_idx in range(CONFIG["n_genes"]):
                for seed in seeds:
                    records.append({
                        "Split": split_name,
                        "Condition": cond,
                        "Model": "Linear (Ridge)",
                        "Seed": seed,
                        "Gene_Idx": g_idx,
                        "Gene_Name": gene_names[g_idx] if g_idx < len(gene_names) else f"gene_{g_idx}",
                        "True_Value": float(trues[i, g_idx]),
                        "Pred_Value": float(preds[i, g_idx]),
                        "Preprocessing": CONFIG["preprocessing"],
                        "Target_Type": CONFIG["target_type"],
                    })
    return records


def main():
    print("=" * 70)
    print("GCP-Mamba Canonical Results Pipeline")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    device = CONFIG["device"]

    # Record input checksums
    checksums = {
        "h5ad": compute_file_checksum(CONFIG["h5ad_path"]),
        "splits": compute_file_checksum(CONFIG["splits_path"]),
        "config": CONFIG,
    }

    # Initialize data
    print("\nInitializing DataEngine...")
    engine = DataEngine(
        top_genes=CONFIG["n_genes"],
        h5ad_path=CONFIG["h5ad_path"],
        splits_path=CONFIG["splits_path"]
    )
    engine.prepare_data()
    gene_names = engine.gene_names
    D = engine.D.to(device)

    # Permuted distance matrix
    torch.manual_seed(0)
    perm_idx = torch.randperm(CONFIG["n_genes"])
    D_permuted = D[perm_idx][:, perm_idx]

    # Constant distance matrix (control)
    D_constant = torch.ones_like(D) * 0.5
    D_constant.fill_diagonal_(0.0)

    # Define splits
    splits = [
        ("Seen 2/2", engine.seen2_loader),
        ("Seen 1/2", engine.seen1_loader),
        ("Seen 0/2", engine.seen0_loader),
    ]

    all_records = []
    loss_history = {}

    # ── Deterministic baselines (seed-independent) ──
    print("\n--- Computing deterministic baselines ---")
    for split_name, loader in splits:
        if len(loader.dataset) == 0:
            continue
        all_records.extend(build_additive_records(
            engine, gene_names, split_name, loader, CONFIG["seeds"]))
        all_records.extend(build_condition_mean_records(
            engine, gene_names, split_name, loader, CONFIG["seeds"]))
        all_records.extend(build_linear_records(
            engine, gene_names, split_name, loader, CONFIG["seeds"]))
    print(f"  Baselines: {len(all_records)} records")

    # ── Neural models (seed-dependent) ──
    for seed in CONFIG["seeds"]:
        print(f"\n=== Seed {seed} ===")
        torch.manual_seed(seed)
        np.random.seed(seed)

        models = {
            "BaseMamba": BaseMamba(
                n_genes=CONFIG["n_genes"],
                d_model=CONFIG["d_model"],
                n_layers=CONFIG["n_layers"]
            ).to(device),
            "GCP-Mamba": GCPMamba(
                n_genes=CONFIG["n_genes"], D=D,
                d_model=CONFIG["d_model"],
                n_layers=CONFIG["n_layers"],
                constrained=CONFIG["constrained_monotonic"]
            ).to(device),
            "GCP-Mamba (Permuted)": GCPMamba(
                n_genes=CONFIG["n_genes"], D=D_permuted,
                d_model=CONFIG["d_model"],
                n_layers=CONFIG["n_layers"],
                constrained=CONFIG["constrained_monotonic"]
            ).to(device),
            "GCP-Mamba (Constant D)": GCPMamba(
                n_genes=CONFIG["n_genes"], D=D_constant,
                d_model=CONFIG["d_model"],
                n_layers=CONFIG["n_layers"],
                constrained=CONFIG["constrained_monotonic"]
            ).to(device),
        }

        loss_history[seed] = {}
        for name, model in models.items():
            print(f"  Training {name}...")
            optimizer = optim.AdamW(
                model.parameters(),
                lr=CONFIG["lr"],
                weight_decay=CONFIG["weight_decay"]
            )

            loss_history[seed][name] = []
            for ep in range(1, CONFIG["epochs"] + 1):
                loss = train_one_epoch(model, engine.train_loader, optimizer, device)
                loss_history[seed][name].append(loss)
                if ep % 25 == 0 or ep == CONFIG["epochs"]:
                    print(f"    Epoch {ep:3d} | loss={loss:.6f}")

            # Evaluate on all splits
            for split_name, loader in splits:
                if len(loader.dataset) == 0:
                    continue
                records = evaluate_to_records(
                    model, loader, name, seed, split_name, gene_names, device
                )
                all_records.extend(records)

    # ── Save canonical predictions ──
    print(f"\nTotal records: {len(all_records)}")
    df = pd.DataFrame(all_records)
    df.to_csv(CONFIG["output_file"], index=False)

    # Compute output checksum
    checksums["output_sha256"] = compute_file_checksum(CONFIG["output_file"])
    checksums["n_records"] = len(all_records)
    checksums["timestamp"] = datetime.now().isoformat()

    with open("canonical_checksums.json", "w") as f:
        json.dump(checksums, f, indent=2)

    with open("canonical_losses.json", "w") as f:
        json.dump(loss_history, f)

    print(f"\nSaved {CONFIG['output_file']} ({len(all_records)} records)")
    print(f"Saved canonical_checksums.json")
    print(f"Saved canonical_losses.json")
    print(f"Output SHA256: {checksums['output_sha256'][:16]}...")


if __name__ == "__main__":
    main()
