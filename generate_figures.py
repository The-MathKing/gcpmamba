import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import json
import scipy.stats as st

def calc_metrics_per_condition(df, top_k=20):
    # df has columns: Split, Condition, Model, Seed, Gene_Idx, True_Value, Pred_Value
    results = []
    
    # Group by everything except Gene_Idx
    groups = df.groupby(["Split", "Condition", "Model", "Seed"])
    for (split, cond, model, seed), group in groups:
        yt = group['True_Value'].values
        yp = group['Pred_Value'].values
        
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
            
        results.append({
            "Split": split,
            "Condition": cond,
            "Model": model,
            "Seed": seed,
            "MSE": mse,
            "Pearson": r
        })
        
    return pd.DataFrame(results)

def generate_figures():
    print("Loading predictions.csv...")
    df_raw = pd.read_csv("predictions.csv")
    
    print("Calculating condition-level metrics...")
    df_metrics = calc_metrics_per_condition(df_raw, top_k=20)
    
    # We want to aggregate over Condition AND Seed to get final model performance per split.
    # Then aggregate over Seed (mean and std dev / CI across seeds)
    df_seed_agg = df_metrics.groupby(["Split", "Model", "Seed"])[["MSE", "Pearson"]].mean().reset_index()
    
    # Then aggregate over Seed (mean and std dev / CI across seeds)
    df_final = df_seed_agg.groupby(["Split", "Model"]).agg(
        MSE_mean=("MSE", "mean"),
        MSE_std=("MSE", "std"),
        Pearson_mean=("Pearson", "mean"),
        Pearson_std=("Pearson", "std"),
        n_seeds=("Seed", "count")
    ).reset_index()
    
    # Calculate 95% CI
    t_val = st.t.ppf(0.975, df_final['n_seeds'] - 1)
    df_final['MSE_ci'] = t_val * df_final['MSE_std'] / np.sqrt(df_final['n_seeds'])
    df_final['Pearson_ci'] = t_val * df_final['Pearson_std'] / np.sqrt(df_final['n_seeds'])
    
    print("\n=== FINAL RESULTS ===")
    print(df_final.to_string(index=False))
    
    df_final.to_csv("final_metrics.csv", index=False)
    
    # Generate Bar Chart
    sns.set_theme(style="whitegrid", font_scale=1.1)
    
    splits = ["Seen 2/2", "Seen 1/2", "Seen 0/2"]
    
    # Filter to only the main models for the chart
    models_to_plot = ["Additive", "Condition Mean", "Linear", "GEARS", "BaseMamba", "GCP-Mamba (Permuted GO)", "GCP-Mamba"]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for i, metric in enumerate(["MSE_mean", "Pearson_mean"]):
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
        ax.set_title("MSE (All Genes)" if metric == "MSE_mean" else "Pearson Correlation (Top 20 DEGs)")
        if i == 0:
            ax.legend()
            
    plt.tight_layout()
    plt.savefig("benchmarking_results.png", dpi=300)
    print("Saved benchmarking_results.png")

if __name__ == "__main__":
    generate_figures()
