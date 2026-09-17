import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import json
import scipy.stats as st

plt.rcParams.update({
    'font.size': 20, 
    'axes.titlesize': 24, 
    'axes.labelsize': 20, 
    'xtick.labelsize': 18, 
    'ytick.labelsize': 18, 
    'legend.fontsize': 18
})

def calc_metrics_per_condition(df, df_additive, top_k=20):
    results = []
    
    # Pre-build additive dictionary for fast lookup: (Split_ID, Split, Condition, Seed, Gene_Idx) -> Additive Pred_Value
    add_dict = {}
    for _, row in df_additive.iterrows():
        add_dict[(row.get('Split_ID', 0), row['Condition'], row['Seed'], row['Gene_Idx'])] = row['Pred_Value']
    
    # Group by everything except Gene_Idx
    groups = df.groupby(["Split_ID", "Split", "Condition", "Model", "Seed"])
    for (split_id, split, cond, model, seed), group in groups:
        group = group.sort_values("Gene_Idx")
        yt = group['True_Value'].values
        yp = group['Pred_Value'].values
        genes = group['Gene_Idx'].values
        
        mse = np.mean((yt - yp)**2)
        
        idx = np.argsort(np.abs(yt))[-top_k:]
        yt_k = yt[idx]
        yp_k = yp[idx]
        
        if np.std(yt_k) > 1e-6 and np.std(yp_k) > 1e-6:
            r, _ = pearsonr(yt_k, yp_k)
        else:
            r = 0.0
            
        yt_synergy = np.zeros(len(yt))
        yp_synergy = np.zeros(len(yt))
        valid_synergy = True
        for i, g in enumerate(genes):
            key = (split_id, cond, seed, g)
            if key in add_dict:
                add_pred = add_dict[key]
                yt_synergy[i] = yt[i] - add_pred
                yp_synergy[i] = yp[i] - add_pred
        
        synergy_magnitude = np.mean(np.abs(yt_synergy))
                
        if model in ["Additive", "Condition Mean", "Linear"]:
            r_syn = np.nan
        elif valid_synergy and '+' in cond and np.std(yt_synergy) > 1e-6 and np.std(yp_synergy) > 1e-6:
            r_syn, _ = pearsonr(yt_synergy, yp_synergy)
        else:
            r_syn = np.nan
            
        results.append({
            "Split_ID": split_id,
            "Split": split,
            "Condition": cond,
            "Model": model,
            "Seed": seed,
            "MSE": mse,
            "Pearson": r,
            "Synergy_Pearson": r_syn,
            "Synergy_Magnitude": synergy_magnitude
        })
        
    df_res = pd.DataFrame(results)
    
    df_doubles = df_res[df_res['Condition'].str.contains('\+')]
    if len(df_doubles) > 0:
        median_mag = df_doubles['Synergy_Magnitude'].median()
        df_res['Stratum'] = df_res.apply(lambda row: 'High Synergy' if row['Synergy_Magnitude'] >= median_mag else 'Low Synergy', axis=1)
    else:
        df_res['Stratum'] = 'All'
        
    return df_res

def bootstrap_ci(data, n_bootstraps=1000, alpha=0.05):
    """Compute bootstrap CI for the mean."""
    if len(data) == 0:
        return 0.0, 0.0
    data = np.array(data)
    bootstrapped_means = np.zeros(n_bootstraps)
    for i in range(n_bootstraps):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrapped_means[i] = np.mean(sample)
    lower = np.percentile(bootstrapped_means, 100 * (alpha / 2))
    upper = np.percentile(bootstrapped_means, 100 * (1 - alpha / 2))
    return lower, upper

def generate_figures():
    print("Loading canonical_predictions.csv...")
    try:
        df_raw = pd.read_csv("canonical_predictions.csv")
    except FileNotFoundError:
        df_raw = pd.read_csv("predictions.csv")
        
    if 'Split_ID' not in df_raw.columns:
        df_raw['Split_ID'] = 0
        
    df_additive = df_raw[df_raw['Model'] == 'Additive']
    
    print("Calculating condition-level metrics...")
    df_metrics = calc_metrics_per_condition(df_raw, df_additive, top_k=20)
    
    # We aggregate over Condition, Seed, Split_ID
    # But we want to pool across Split_IDs and Conditions for the final evaluation
    # So we group by Split, Model, Seed to get the mean for each seed, then across seeds
    # Wait, the reviewer asked for condition-level bootstrap CIs on the pooled conditions.
    
    # Group by Split, Model to get all conditions across all seeds and splits
    # We want to treat each (Condition, Split_ID) as an independent observation for the condition-level evaluation.
    # To get the prediction for a condition, we can average over the 3 seeds first.
    
    df_seed_mean = df_metrics.groupby(["Split_ID", "Split", "Condition", "Model", "Stratum"]).agg(
        MSE=("MSE", "mean"),
        Pearson=("Pearson", "mean"),
        Synergy_Pearson=("Synergy_Pearson", "mean")
    ).reset_index()
    
    df_final_list = []
    
    # For each split and model, pool all (Condition, Split_ID) pairs
    groups = df_seed_mean.groupby(["Split", "Model"])
    for (split, model), group in groups:
        n_conditions = len(group)
        mse_vals = group["MSE"].dropna().values
        pearson_vals = group["Pearson"].dropna().values
        synergy_vals = group["Synergy_Pearson"].dropna().values
        
        mse_mean = np.mean(mse_vals) if len(mse_vals) > 0 else 0
        pearson_mean = np.mean(pearson_vals) if len(pearson_vals) > 0 else 0
        synergy_mean = np.mean(synergy_vals) if len(synergy_vals) > 0 else 0
        
        mse_ci_l, mse_ci_u = bootstrap_ci(mse_vals)
        pearson_ci_l, pearson_ci_u = bootstrap_ci(pearson_vals)
        synergy_ci_l, synergy_ci_u = bootstrap_ci(synergy_vals)
        
        # Approximate CI half-width for backward compatibility with plotting script
        mse_ci = (mse_ci_u - mse_ci_l) / 2.0
        pearson_ci = (pearson_ci_u - pearson_ci_l) / 2.0
        synergy_ci = (synergy_ci_u - synergy_ci_l) / 2.0
        
        df_final_list.append({
            "Split": split,
            "Model": model,
            "MSE_mean": mse_mean,
            "Pearson_mean": pearson_mean,
            "Synergy_mean": synergy_mean,
            "MSE_ci": mse_ci,
            "Pearson_ci": pearson_ci,
            "Synergy_ci": synergy_ci,
            "n_conditions": n_conditions
        })
        
    df_final = pd.DataFrame(df_final_list)
    
    print("\n=== FINAL RESULTS (Pooled Conditions) ===")
    print(df_final.to_string(index=False))
    df_final.to_csv("final_metrics.csv", index=False)
    
    print("\n=== ABLATION TABLE (NicheDeSig Table 4 Style) ===")
    df_stratum_final = df_seed_mean.groupby(["Split", "Stratum", "Model"]).agg(
        MSE=("MSE", "mean"),
        Pearson=("Pearson", "mean"),
        Synergy=("Synergy_Pearson", "mean")
    ).reset_index()
    
    print(df_stratum_final.to_string(index=False))
    df_stratum_final.to_csv("ablation_table_stratified.csv", index=False)
    
    # Perform Paired t-test for Synergy Metric: GCP-Mamba vs GCP-Mamba (Permuted GO)
    print("\n--- Statistical Test: Synergy Recovery (Seen 0/2) ---")
    df_seen0 = df_seed_mean[df_seed_mean["Split"] == "Seen 0/2"]
    
    # Ensure they are aligned by Condition and Split_ID
    gcp_data = df_seen0[df_seen0["Model"] == "GCP-Mamba"].set_index(["Split_ID", "Condition"])["Synergy_Pearson"]
    perm_data = df_seen0[df_seen0["Model"] == "GCP-Mamba (Permuted)"].set_index(["Split_ID", "Condition"])["Synergy_Pearson"]
    
    # Merge to align exactly
    aligned = pd.merge(gcp_data, perm_data, left_index=True, right_index=True, suffixes=('_gcp', '_perm'))
    aligned = aligned.dropna()
    
    gcp_syn = aligned["Synergy_Pearson_gcp"].values
    perm_syn = aligned["Synergy_Pearson_perm"].values
    
    if len(gcp_syn) > 0 and len(perm_syn) > 0:
        t_stat, p_val = st.ttest_rel(gcp_syn, perm_syn)
        print(f"Paired t-test (N={len(gcp_syn)} pooled conditions): t={t_stat:.3f}, p={p_val:.4e}")
    else:
        print("Could not compute t-test.")

if __name__ == "__main__":
    generate_figures()
