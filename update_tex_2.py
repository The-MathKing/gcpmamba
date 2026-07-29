import pandas as pd

def main():
    try:
        df = pd.read_csv("final_metrics.csv")
    except Exception as e:
        print(f"Error reading final_metrics.csv: {e}")
        return
        
    with open('gcp_mamba_plan.tex', 'r') as f:
        content = f.read()

    # Models: Condition Mean, Additive, Linear, GEARS, BaseMamba, GCP-Mamba (Permuted GO), GCP-Mamba
    # TBD placeholders:
    # TBD_MEAN_MSE, TBD_MEAN_R
    # TBD_ADD_MSE, TBD_ADD_R
    # TBD_LIN_MSE, TBD_LIN_R
    # TBD_GEARS_MSE, TBD_GEARS_R
    # TBD_BASE_MSE, TBD_BASE_R
    # TBD_PERM_MSE, TBD_PERM_R
    # TBD_GCP_MSE, TBD_GCP_R
    
    mapping = {
        "Condition Mean": ("TBD_MEAN_MSE", "TBD_MEAN_R"),
        "Additive": ("TBD_ADD_MSE", "TBD_ADD_R"),
        "Linear": ("TBD_LIN_MSE", "TBD_LIN_R"),
        "GEARS": ("TBD_GEARS_MSE", "TBD_GEARS_R"),
        "BaseMamba": ("TBD_BASE_MSE", "TBD_BASE_R"),
        "GCP-Mamba (Permuted GO)": ("TBD_PERM_MSE", "TBD_PERM_R"),
        "GCP-Mamba": ("TBD_GCP_MSE", "TBD_GCP_R")
    }
    
    # We focus on the "Seen 0/2" split for the table
    df_02 = df[df['Split'] == 'Seen 0/2']
    
    for model_name, (mse_tag, r_tag) in mapping.items():
        row = df_02[df_02['Model'] == model_name]
        if not row.empty:
            mse_str = f"${row['MSE_mean'].values[0]:.4f} \\pm {row['MSE_ci'].values[0]:.4f}$"
            r_str = f"${row['Pearson_mean'].values[0]:.3f} \\pm {row['Pearson_ci'].values[0]:.3f}$"
            content = content.replace(mse_tag, mse_str)
            content = content.replace(r_tag, r_str)
            
    with open('gcp_mamba_plan.tex', 'w') as f:
        f.write(content)

    print("Updated gcp_mamba_plan.tex successfully.")

if __name__ == '__main__':
    main()
