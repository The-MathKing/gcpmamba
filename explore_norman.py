import scanpy as sc
import numpy as np
import json

def explore():
    print("Loading Norman dataset...")
    adata = sc.read_h5ad('data/norman/perturb_processed.h5ad')
    
    conditions = adata.obs['condition'].unique()
    
    # Singles end with +ctrl (e.g. 'KLF1+ctrl')
    # Let's clean the '+ctrl' part to just get the gene name
    singles_raw = [c for c in conditions if c.endswith('+ctrl') and c != 'ctrl']
    single_genes = [c.split('+')[0] for c in singles_raw]
    
    doubles = [c for c in conditions if '+' in c and 'ctrl' not in c]
    
    print(f"Total cells: {adata.shape[0]}")
    print(f"Single genes: {len(single_genes)}")
    print(f"Double conditions: {len(doubles)}")
    
    # Let's collect all genes involved in doubles
    genes_in_doubles = set()
    for c in doubles:
        g1, g2 = c.split('+')
        genes_in_doubles.add(g1)
        genes_in_doubles.add(g2)
        
    print(f"Genes involved in doubles: {len(genes_in_doubles)}")
    
    # The held-out set should ideally be selected from genes_in_doubles so that holding them out actually affects the Seen splits!
    valid_candidates = list(genes_in_doubles)
    
    rng = np.random.default_rng(42)
    held_out_15 = rng.choice(valid_candidates, 15, replace=False).tolist()
    
    print(f"Held out genes (15): {held_out_15}")
    
    seen2 = []
    seen1 = []
    seen0 = []
    
    for c in doubles:
        g1, g2 = c.split('+')
        in_1 = g1 in held_out_15
        in_2 = g2 in held_out_15
        if in_1 and in_2:
            seen0.append(c)
        elif in_1 or in_2:
            seen1.append(c)
        else:
            seen2.append(c)
            
    print(f"Doubles in Seen 2/2: {len(seen2)}")
    print(f"Doubles in Seen 1/2: {len(seen1)}")
    print(f"Doubles in Seen 0/2: {len(seen0)}")
    
    # We also need to define the train singles.
    # If a gene is in held_out_15, its single perturbation is NOT in the training set.
    train_singles = [c for c in singles_raw if c.split('+')[0] not in held_out_15]
    
    manifest = {
        "held_out_genes": held_out_15,
        "train_singles": train_singles,
        "test_singles": [c for c in singles_raw if c.split('+')[0] in held_out_15],
        "seen2_doubles": seen2,
        "seen1_doubles": seen1,
        "seen0_doubles": seen0
    }
    
    with open('splits_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
        
    print("Saved splits_manifest.json")

if __name__ == '__main__':
    explore()
