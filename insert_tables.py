import pandas as pd
import json

def generate_table_3():
    with open('realism_check_results.json', 'r') as f:
        res = json.load(f)
    
    tex = r"""\begin{table}[!htbp]
\centering
\caption{Realism Check: Held-out Target Genes vs. Training Distribution}
\label{tab:realism}
\begin{tabular}{lll}
\toprule
\textbf{Metric} & \textbf{KS-Statistic} & \textbf{p-value} \\
\midrule
Expression Magnitude & """ + f"{res['ks_mean_stat']:.4f} & {res['ks_mean_p']:.4e}" + r""" \\
Expression Variance & """ + f"{res['ks_var_stat']:.4f} & {res['ks_var_p']:.4e}" + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""
    return tex

def generate_table_4():
    df = pd.read_csv('ablation_table_stratified.csv')
    # Filter for Seen 0/2 for simplicity
    df = df[df['Split'] == 'Seen 0/2']
    
    tex = r"""\begin{table*}[!htbp]
\centering
\caption{Stratified Ablation Analysis (Seen 0/2 Zero-Shot Extrapolation)}
\label{tab:ablation}
\begin{tabular}{llccc}
\toprule
\textbf{Model} & \textbf{Stratum} & \textbf{MSE} & \textbf{Pearson $r$} & \textbf{Synergy $r$} \\
\midrule
"""
    for _, row in df.iterrows():
        model = row['Model'].replace('&', '\&').replace('_', '\_')
        stratum = row['Stratum']
        mse = row['MSE']
        pearson = row['Pearson']
        synergy = row['Synergy']
        if pd.isna(synergy):
            synergy_str = "---"
        else:
            synergy_str = f"{synergy:.3f}"
            
        tex += f"{model} & {stratum} & {mse:.4f} & {pearson:.3f} & {synergy_str} \\\\\n"
        
    tex += r"""\bottomrule
\end{tabular}
\end{table*}
"""
    return tex

def generate_table_5():
    with open('convergence_results.json', 'r') as f:
        res = json.load(f)
        
    tex = r"""\begin{table}[!htbp]
\centering
\caption{Convergence Stability Across Learning Rates}
\label{tab:convergence}
\begin{tabular}{lll}
\toprule
\textbf{Learning Rate} & \textbf{Final Loss} & \textbf{Late-Stage Variance} \\
\midrule
"""
    for lr, data in res.items():
        fl = data['final_loss']
        v = data['variance']
        tex += f"{float(lr):.0e} & {fl:.4f} & {v:.2e} \\\\\n"
        
    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return tex

def main():
    with open('gcp_mamba_plan.tex', 'r') as f:
        content = f.read()
        
    # We will inject these tables before the biological case study
    table_3 = generate_table_3()
    table_4 = generate_table_4()
    table_5 = generate_table_5()
    
    marker = r"\subsection{Biological Case Study: Co-expression Topology Driven State Retention}"
    
    injection = f"\n{table_3}\n{table_4}\n{table_5}\n"
    
    new_content = content.replace(marker, injection + "\n" + marker)
    
    with open('gcp_mamba_plan_updated.tex', 'w') as f:
        f.write(new_content)
        
    print("Tables generated and injected into gcp_mamba_plan_updated.tex")

if __name__ == "__main__":
    main()
