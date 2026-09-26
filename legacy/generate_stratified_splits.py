import scanpy as sc
import json
import numpy as np
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_splits", type=int, default=1)
    args = parser.parse_args()
    
    h5ad_path = 'data/norman/perturb_processed.h5ad'
    print(f"Loading {h5ad_path}...")
    adata = sc.read_h5ad(h5ad_path)
    
    # Calculate gene means
    gene_means = np.array(adata.X.mean(axis=0)).flatten()
    gene_names = adata.var['gene_name'].values
    
    # Get all perturbed genes
    conditions = [c for c in adata.obs['condition'].unique() if c != 'ctrl']
    pert_genes = []
    for c in conditions:
        parts = c.split('+')
        for p in parts:
            if p != 'ctrl' and p not in pert_genes:
                pert_genes.append(p)
                
    # Keep only those in the 500-gene subset
    valid_pert_genes = pert_genes
    
    print(f"Found {len(valid_pert_genes)} valid perturbed genes.")
    
    # Bin genes into quartiles by expression mean
    mean_dict = dict(zip(gene_names, gene_means))
    valid_means = [mean_dict[g] for g in valid_pert_genes if g in mean_dict]
    valid_pert_genes = [g for g in valid_pert_genes if g in mean_dict]
    
    q25, q50, q75 = np.percentile(valid_means, [25, 50, 75])
    
    bins = [[], [], [], []]
    for g, m in zip(valid_pert_genes, valid_means):
        if m < q25: bins[0].append(g)
        elif m < q50: bins[1].append(g)
        elif m < q75: bins[2].append(g)
        else: bins[3].append(g)
        
    np.random.seed(42)
    
    for split_idx in range(args.n_splits):
        # Pick 15 genes stratified (approx 4, 4, 4, 3)
        test_genes = []
        test_genes.extend(np.random.choice(bins[0], 4, replace=False))
        test_genes.extend(np.random.choice(bins[1], 4, replace=False))
        test_genes.extend(np.random.choice(bins[2], 4, replace=False))
        test_genes.extend(np.random.choice(bins[3], 3, replace=False))
        
        train_singles = []
        test_singles = []
        seen2_doubles = []
        seen1_doubles = []
        seen0_doubles = []
        
        for c in conditions:
            parts = c.split('+')
            if len(parts) == 1 or 'ctrl' in parts:
                gene = parts[0] if parts[0] != 'ctrl' else parts[1]
                if gene in test_genes:
                    test_singles.append(c)
                else:
                    train_singles.append(c)
            elif len(parts) == 2:
                g1, g2 = parts
                c1 = g1 in test_genes
                c2 = g2 in test_genes
                if c1 and c2:
                    seen0_doubles.append(c)
                elif c1 or c2:
                    seen1_doubles.append(c)
                else:
                    seen2_doubles.append(c)
                    
        manifest = {
            "train_singles": train_singles,
            "test_singles": test_singles,
            "seen2_doubles": seen2_doubles,
            "seen1_doubles": seen1_doubles,
            "seen0_doubles": seen0_doubles
        }
        
        out_path = f"splits/stratified_manifest_{split_idx}.json"
        with open(out_path, 'w') as f:
            json.dump(manifest, f, indent=4)
        print(f"Saved {out_path} with {len(seen0_doubles)} Seen 0/2 conditions")
        
if __name__ == "__main__":
    main()
