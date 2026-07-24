import os
import scanpy as sc

def main():
    h5ad_path = './data/norman/perturb_processed.h5ad'
    if not os.path.exists(h5ad_path):
        print(f"File not found: {h5ad_path}")
        return
        
    print(f"Loading {h5ad_path} via Scanpy...")
    adata = sc.read_h5ad(h5ad_path)
    
    print(f"Dataset loaded successfully!")
    print(f"Shape (Cells, Genes): {adata.shape}")
    
    # Inspect conditions
    conditions = adata.obs['condition'].unique()
    print(f"Number of unique perturbation conditions: {len(conditions)}")
    
    # Check if 'ctrl' is there
    ctrl_cells = adata[adata.obs['condition'] == 'ctrl']
    print(f"Number of control cells: {len(ctrl_cells)}")
    
    # Check top genes
    genes = adata.var_names
    print(f"First 10 genes: {genes[:10].tolist()}")

if __name__ == "__main__":
    main()
