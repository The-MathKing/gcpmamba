import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import json
import scipy.stats as st

def calc_metrics_per_condition(df, df_additive, top_k=20):
    # df has columns: Split, Condition, Model, Seed, Gene_Idx, True_Value, Pred_Value
    results = []
    
    # Pre-build additive dictionary for fast lookup: (Split, Condition, Seed, Gene_Idx) -> Additive Pred_Value
    # Note: Additive true_value is the same as the target. Additive prediction is the additive baseline.
    add_dict = {}
    for _, row in df_additive.iterrows():
        add_dict[(row['Condition'], row['Seed'], row['Gene_Idx'])] = row['Pred_Value']
    
    # Group by everything except Gene_Idx
    groups = df.groupby(["Split", "Condition", "Model", "Seed"])
    for (split, cond, model, seed), group in groups:
        # sort by Gene_Idx to align
        group = group.sort_values("Gene_Idx")
        yt = group['True_Value'].values
        yp = group['Pred_Value'].values
        genes = group['Gene_Idx'].values
        
        # Calculate MSE over all genes
        mse = np.mean((yt - yp)**2)
        
        # Calculate Pearson over top_k differentially expressed genes
        idx = np.argsort(np.abs(yt))[-top_k:]
        yt_k = yt[idx]
        yp_k = yp[idx]
        
        if np.std(yt_k) > 1e-6 and np.std(yp_k) > 1e-6:
            r, _ = pearsonr(yt_k, yp_k)
        else:
            r = 0.0
            
        # --- Synergy Metric ---
        # Residual = Target - Additive
        # Pred Residual = Pred - Additive
        yt_synergy = np.zeros(len(yt))
        yp_synergy = np.zeros(len(yt))
        valid_synergy = True
        for i, g in enumerate(genes):
            if (cond, seed, g) in add_dict:
                add_pred = add_dict[(cond, seed, g)]
                yt_synergy[i] = yt[i] - add_pred
                yp_synergy[i] = yp[i] - add_pred
            else:
                valid_synergy = False
                break
                
        # Only compute synergy correlation if it's a combinatorial condition (has a valid additive baseline)
        if valid_synergy and '+' in cond and np.std(yt_synergy) > 1e-6 and np.std(yp_synergy) > 1e-6:
            r_syn, _ = pearsonr(yt_synergy, yp_synergy)
        else:
            r_syn = np.nan
            
        results.append({
            "Split": split,
            "Condition": cond,
            "Model": model,
            "Seed": seed,
            "MSE": mse,
            "Pearson": r,
            "Synergy_Pearson": r_syn
        })
        
    return pd.DataFrame(results)

def generate_figures():
    print("Loading predictions.csv...")
    df_raw = pd.read_csv("predictions.csv")
    
    # Extract additive predictions
    df_additive = df_raw[df_raw['Model'] == 'Additive']
    
    print("Calculating condition-level metrics...")
    df_metrics = calc_metrics_per_condition(df_raw, df_additive, top_k=20)
    
    # We want to aggregate over Condition AND Seed to get final model performance per split.
    df_seed_agg = df_metrics.groupby(["Split", "Model", "Seed"])[["MSE", "Pearson", "Synergy_Pearson"]].mean(numeric_only=True).reset_index()
    
    # Then aggregate over Seed (mean and std dev / CI across seeds)
    df_final = df_seed_agg.groupby(["Split", "Model"]).agg(
        MSE_mean=("MSE", "mean"),
        MSE_std=("MSE", "std"),
        Pearson_mean=("Pearson", "mean"),
        Pearson_std=("Pearson", "std"),
        Synergy_mean=("Synergy_Pearson", "mean"),
        Synergy_std=("Synergy_Pearson", "std"),
        n_seeds=("Seed", "count")
    ).reset_index()
    
    # Calculate 95% CI
    t_val = st.t.ppf(0.975, df_final['n_seeds'] - 1)
    df_final['MSE_ci'] = t_val * df_final['MSE_std'] / np.sqrt(df_final['n_seeds'])
    df_final['Pearson_ci'] = t_val * df_final['Pearson_std'] / np.sqrt(df_final['n_seeds'])
    df_final['Synergy_ci'] = t_val * df_final['Synergy_std'] / np.sqrt(df_final['n_seeds'])
    
    print("\n=== FINAL RESULTS ===")
    print(df_final.to_string(index=False))
    
    df_final.to_csv("final_metrics.csv", index=False)
    
    # Generate Bar Chart
    sns.set_theme(style="whitegrid", font_scale=1.1)
    
    splits = ["Seen 2/2", "Seen 1/2", "Seen 0/2"]
    
    # Filter to only the main models for the chart
    models_to_plot = ["Additive", "Condition Mean", "Linear", "GEARS", "BaseMamba", "GCP-Mamba (Permuted GO)", "GCP-Mamba"]
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for i, metric in enumerate(["MSE_mean", "Pearson_mean", "Synergy_mean"]):
        ax = axes[i]
        
        x = np.arange(len(splits))
        width = 0.8 / len(models_to_plot)
        
        for j, model in enumerate(models_to_plot):
            model_data = df_final[df_final["Model"] == model]
            
            means = []
            cis = []
            for split in splits:
                row = model_data[model_data["Split"] == split]
                if not row.empty:
                    means.append(row[metric].values[0])
                    cis.append(row[metric.replace("_mean", "_ci")].values[0])
                else:
                    means.append(0)
                    cis.append(0)
                    
            offset = (j - len(models_to_plot)/2) * width + width/2
            ax.bar(x + offset, means, width, label=model, yerr=cis, capsize=3)
            
        ax.set_xticks(x)
        ax.set_xticklabels(splits)
        
        if metric == "MSE_mean":
            ax.set_title("MSE (All Genes)")
        elif metric == "Pearson_mean":
            ax.set_title("Pearson Correlation (Top 20 DEGs)")
        else:
            ax.set_title("Synergy Recovery (Residual Correlation)")
            
        if i == 0:
            ax.legend(fontsize=10)
            
    plt.tight_layout()
    plt.savefig("benchmarking_results.png", dpi=300)
    print("Saved benchmarking_results.png")
    
    # Perform Paired t-test for Synergy Metric: GCP-Mamba vs GCP-Mamba (Permuted GO)
    print("\n--- Statistical Test: Synergy Recovery (Seen 0/2) ---")
    df_seen0 = df_seed_agg[df_seed_agg["Split"] == "Seen 0/2"]
    gcp_syn = df_seen0[df_seen0["Model"] == "GCP-Mamba"]["Synergy_Pearson"].values
    perm_syn = df_seen0[df_seen0["Model"] == "GCP-Mamba (Permuted GO)"]["Synergy_Pearson"].values
    
    if len(gcp_syn) > 0 and len(perm_syn) > 0:
        t_stat, p_val = st.ttest_rel(gcp_syn, perm_syn)
        print(f"Paired t-test (N={len(gcp_syn)} seeds): t={t_stat:.3f}, p={p_val:.4e}")
    else:
        print("Could not compute t-test.")

if __name__ == "__main__":
    generate_figures()
