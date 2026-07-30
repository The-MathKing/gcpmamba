import scanpy as sc
import numpy as np
import json
import scipy.stats as st

def main():
    print("Loading Norman dataset...")
    adata = sc.read_h5ad('data/norman/perturb_processed.h5ad')
    
    with open('splits_manifest.json', 'r') as f:
        manifest = json.load(f)
        
    held_out_genes = manifest['held_out_genes']
    
    # Do not subset to HVGs so we don't lose the held-out genes.
    
    # We want to compare the distribution of expression magnitude (mean) and variance
    # between the held-out target genes and the rest of the 500 HVGs
    
    if 'gene_name' in adata.var:
        all_genes = list(adata.var['gene_name'])
    else:
        all_genes = list(adata.var_names)
    
    held_out_idx = [i for i, g in enumerate(all_genes) if g in held_out_genes]
    train_idx = [i for i, g in enumerate(all_genes) if g not in held_out_genes]
    print(f"Held out mapped: {len(held_out_idx)}")
    print(f"Sample held out genes: {held_out_genes[:5]}")
    print(f"Sample var_names: {all_genes[:5]}")
    
    # Calculate means and variances across cells
    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
        
    means = np.mean(X, axis=0)
    variances = np.var(X, axis=0)
    
    train_means = means[train_idx]
    held_out_means = means[held_out_idx]
    
    train_vars = variances[train_idx]
    held_out_vars = variances[held_out_idx]
    
    # KS Test
    ks_mean_stat, p_mean = st.ks_2samp(train_means, held_out_means)
    ks_var_stat, p_var = st.ks_2samp(train_vars, held_out_vars)
    
    print("\n=== Realism Check: Held-out vs Training Gene Distributions ===")
    print(f"Expression Magnitude (Mean): KS-stat = {ks_mean_stat:.4f}, p-value = {p_mean:.4e}")
    print(f"Expression Variance:         KS-stat = {ks_var_stat:.4f}, p-value = {p_var:.4e}")
    
    res = {
        "ks_mean_stat": float(ks_mean_stat),
        "ks_mean_p": float(p_mean),
        "ks_var_stat": float(ks_var_stat),
        "ks_var_p": float(p_var)
    }
    with open('realism_check_results.json', 'w') as f:
        json.dump(res, f, indent=2)
    print("Saved realism_check_results.json")

if __name__ == "__main__":
    main()
