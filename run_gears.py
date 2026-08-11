import torch
import numpy as np
import pandas as pd
import json
import pickle
import os
import scanpy as sc

# Monkey-patch pandas Series for older scipy/GEARS compatibility
if not hasattr(pd.Series, 'nonzero'):
    pd.Series.nonzero = lambda self: self.to_numpy().nonzero()

from gears import PertData, GEARS

def run_gears_official():
    print("=== Running Official GEARS ===")
    
    print("Initializing PertData to discover valid GO graph conditions...")
    pert_data = PertData('./data')
    pert_data.load(data_name='norman')
    
    # GEARS removes some conditions if they aren't in the GO graph.
    valid_conditions = set(pert_data.dataset_processed.keys())
    
    with open('splits_manifest.json', 'r') as f:
        manifest = json.load(f)
        
    train_conds = manifest['train_singles'] + manifest['seen2_doubles']
    # Filter valid
    train_conds = [c for c in train_conds if c in valid_conditions]
    
    # GEARS requires a validation set for early stopping or tuning. Let's carve out 10% of train for val.
    np.random.seed(42)
    val_size = max(1, int(len(train_conds) * 0.1))
    val_conds = list(np.random.choice(train_conds, val_size, replace=False))
    train_conds = [c for c in train_conds if c not in val_conds]
    
    test_conds = manifest['seen1_doubles'] + manifest['seen0_doubles'] + manifest['test_singles']
    test_conds = [c for c in test_conds if c in valid_conditions]
    
    gears_split = {
        'train': train_conds,
        'val': val_conds,
        'test': test_conds
    }
    
    with open('gears_split_dict.pkl', 'wb') as f:
        pickle.dump(gears_split, f)
        
    print("Setting custom split...")
    pert_data.prepare_split(split='custom', split_dict_path='gears_split_dict.pkl')
    pert_data.get_dataloader(batch_size=32, test_batch_size=128)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    gears_model = GEARS(pert_data, device=device)
    gears_model.model_initialize(hidden_size=64)
    print("Training GEARS...")
    gears_model.train(epochs=20)
    
    # Save predictions
    results_list = []
    
    # GEARS predict() returns predictions and uncertainties
    # We will predict for seen2, seen1, seen0
    
    splits_to_eval = [
        ("Seen 2/2", manifest['seen2_doubles']),
        ("Seen 1/2", manifest['seen1_doubles']),
        ("Seen 0/2", manifest['seen0_doubles'])
    ]
    
    for split_name, conds in splits_to_eval:
        if not conds: continue
        for cond in conds:
            try:
                preds, _ = gears_model.predict([cond])
                # gears_model.predict returns a dict with 'cond' as key, values are (1, N_genes)
                pred_val = preds[cond][0]
                
                # To compare, we need the True value which we can get from the dataloader or adata
                # but to be simple, let's just save the prediction.
                # Actually, our canonical_results.csv needs True_Value.
                # We can extract it from pert_data.adata
                mask = pert_data.adata.obs['condition'] == cond
                if mask.sum() > 0:
                    true_val = pert_data.adata[mask].X.mean(axis=0).A1 # assuming sparse
                else:
                    true_val = np.zeros(len(pred_val))
                
                for g_idx in range(len(pred_val)):
                    results_list.append({
                        "Split": split_name,
                        "Condition": cond,
                        "Model": "GEARS",
                        "Seed": 42,
                        "Gene_Idx": g_idx,
                        "True_Value": float(true_val[g_idx]),
                        "Pred_Value": float(pred_val[g_idx])
                    })
            except Exception as e:
                print(f"Failed on {cond}: {e}")
                
    df = pd.DataFrame(results_list)
    df.to_csv("gears_predictions.csv", index=False)
    print("Saved gears_predictions.csv")

if __name__ == '__main__':
    run_gears_official()
