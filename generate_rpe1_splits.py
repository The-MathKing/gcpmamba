import scanpy as sc
import json
import numpy as np
import os

h5ad_path = 'data/replogle_rpe1_essential/perturb_processed.h5ad'
print(f"Loading {h5ad_path}...")

if not os.path.exists(h5ad_path):
    print(f"Error: {h5ad_path} does not exist.")
    exit(1)

adata = sc.read_h5ad(h5ad_path)

conditions = [c for c in adata.obs['condition'].unique() if c != 'ctrl']

# Separate single vs double perturbations
singles = []
doubles = []

for c in conditions:
    if '+' in c and len(c.split('+')) > 1:
        # In norman, ctrl+gene is a single. In replogle, they might be single genes without +
        parts = c.split('+')
        if 'ctrl' in parts or 'non-targeting' in parts:
            singles.append(c)
        else:
            doubles.append(c)
    else:
        singles.append(c)

print(f"Found {len(singles)} single perturbations and {len(doubles)} double perturbations.")

np.random.seed(42)
np.random.shuffle(singles)

split_idx = int(len(singles) * 0.85)
train_singles = singles[:split_idx]
test_singles = singles[split_idx:]

manifest = {
    "train_singles": train_singles,
    "test_singles": test_singles,
    "seen2_doubles": [],
    "seen1_doubles": [],
    "seen0_doubles": []
}

with open('rpe1_splits.json', 'w') as f:
    json.dump(manifest, f, indent=4)

print(f"Splits saved to rpe1_splits.json: {len(train_singles)} train, {len(test_singles)} test.")
