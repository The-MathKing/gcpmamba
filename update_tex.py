import json
import re

def main():
    try:
        with open('results.json', 'r') as f:
            res = json.load(f)
    except FileNotFoundError:
        res = None
        
    try:
        with open('baseline_results.json', 'r') as f:
            base_res = json.load(f)
    except FileNotFoundError:
        base_res = None
        
    try:
        with open('correlation_results.json', 'r') as f:
            corr_res = json.load(f)
    except FileNotFoundError:
        corr_res = None

    with open('gcp_mamba_plan.tex', 'r') as f:
        content = f.read()

    if res:
        mse = res['gcp_mamba']['Seen 0/2']['mse']
        mse_std = res['gcp_mamba']['Seen 0/2']['mse_std']
        pearson = res['gcp_mamba']['Seen 0/2']['pearson']
        pearson_std = res['gcp_mamba']['Seen 0/2']['pearson_std']
        
        base_mse = res['base_mamba']['Seen 0/2']['mse']
        base_mse_std = res['base_mamba']['Seen 0/2']['mse_std']
        base_pearson = res['base_mamba']['Seen 0/2']['pearson']
        base_pearson_std = res['base_mamba']['Seen 0/2']['pearson_std']
        
        content = re.sub(r'BaseMamba & \$.*?\$ & \$.*?\$', f'BaseMamba & ${base_mse:.4f} \\\\pm {base_mse_std:.4f}$ & ${base_pearson:.3f} \\\\pm {base_pearson_std:.3f}$', content)
        content = re.sub(r'\\textbf\{GCP-Mamba\} & \\textbf\{\$.*?\}\} & \\textbf\{\$.*?\}\}', f'\\\\textbf{{GCP-Mamba}} & \\\\textbf{{${mse:.4f} \\\\pm {mse_std:.4f}$}} & \\\\textbf{{${pearson:.3f} \\\\pm {pearson_std:.3f}$}}', content)

    if base_res:
        gears_mse = base_res['faithful_gcn']['Seen 0/2']['mse']
        gears_mse_std = base_res['faithful_gcn']['Seen 0/2']['mse_std']
        gears_pearson = base_res['faithful_gcn']['Seen 0/2']['pearson']
        gears_pearson_std = base_res['faithful_gcn']['Seen 0/2']['pearson_std']
        content = re.sub(r'GEARS \(Roohani \\textit\{et al\.\}, 2022\) & TBD & TBD', f'GEARS (Roohani \\\\textit{{et al.}}, 2022) & ${gears_mse:.4f} \\\\pm {gears_mse_std:.4f}$ & ${gears_pearson:.3f} \\\\pm {gears_pearson_std:.3f}$', content)
        
        # update the ablation table as well
        content = re.sub(r'GEARS \(GCN\)\s+& TBD & TBD', f'GEARS (GCN)       & ${gears_mse:.4f} \\\\pm {gears_mse_std:.4f}$ & ${gears_pearson:.3f}$', content)

    if corr_res:
        pearson_corr = corr_res['pearson']
        p_val = corr_res['p_value']
        # You could also update text referencing Pearson r for the correlation, e.g. "Pearson r \\approx 0.15"

    with open('gcp_mamba_plan.tex', 'w') as f:
        f.write(content)

    print("Updated gcp_mamba_plan.tex successfully.")

if __name__ == '__main__':
    main()
