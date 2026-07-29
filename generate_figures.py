import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel, t, sem

# ==========================================
# INPUT PROCESSED 10-FOLD DATA HERE
# ==========================================
folds = np.arange(1, 11)
# Replace with your actual 10-fold MSE results (MUST BE RUN TO CONVERGENCE, E.G., 100 EPOCHS)
mse_base = np.array([1.721, 1.765, 1.782, 1.734, 1.801, 1.756, 1.793, 1.768, 1.815, 1.745]) 
mse_gcp = np.array([1.702, 1.741, 1.768, 1.721, 1.780, 1.735, 1.772, 1.750, 1.791, 1.725])
mse_attn_mean = 1.762
mse_attn_std = 0.041

# SOTA Baseline Placeholders (Replace with actual evaluated metrics)
mse_gears_mean = 1.790 # Placeholder
mse_gears_std = 0.030  # Placeholder

def calculate_cohens_d(x, y):
    diff = x - y
    return np.mean(diff) / np.std(diff, ddof=1)

def calculate_confidence_interval(x, y, confidence=0.95):
    diff = x - y
    n = len(diff)
    m = np.mean(diff)
    std_err = sem(diff)
    h = std_err * t.ppf((1 + confidence) / 2., n - 1)
    return m - h, m + h

# ==========================================
# FIGURE 1: 10-Fold Benchmarking Results
# ==========================================
def generate_benchmark_figure():
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.35
    
    ax.bar(folds - width/2, mse_gcp, width, label='GCP-Mamba (w/ $M_{\\Delta}$)', color='#2196F3')
    ax.bar(folds + width/2, mse_base, width, label='BaseMamba (w/o $M_{\\Delta}$)', color='#FF9800')
    
    ax.set_ylabel('MSE (Top-50 Genes)')
    ax.set_xlabel('Cross-Validation Fold')
    ax.set_title('10-Fold Zero-Shot Condition-Level Ablation')
    ax.set_xticks(folds)
    ax.legend()
    
    # Calculate and display p-value, Cohen's d, and 95% CI
    stat, p_val = ttest_rel(mse_gcp, mse_base)
    d = calculate_cohens_d(mse_gcp, mse_base)
    ci_low, ci_high = calculate_confidence_interval(mse_gcp, mse_base)
    
    sig_marker = "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    stat_text = (f"Paired t-test: p = {p_val:.2e} ({sig_marker})\n"
                 f"Cohen's d: {d:.2f}\n"
                 f"95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
    
    plt.figtext(0.15, 0.75, stat_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.savefig('benchmarking_results.png', dpi=300, bbox_inches='tight')
    print("Exported: benchmarking_results.png")

# ==========================================
# FIGURE 2: Architectural Ablation & SOTA
# ==========================================
def generate_ablation_figure():
    # Includes SOTA Baselines
    models = ['GEARS', 'BaseMamba', 'GCP-Mean', 'GCP-Attn']
    mse_means = [mse_gears_mean, np.mean(mse_base), np.mean(mse_gcp), mse_attn_mean]
    mse_stds = [mse_gears_std, np.std(mse_base), np.std(mse_gcp), mse_attn_std]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#F44336', '#9E9E9E', '#4CAF50', '#9C27B0']
    ax.bar(models, mse_means, yerr=mse_stds, capsize=5, color=colors)
    
    ax.set_ylabel('Zero-Shot MSE')
    ax.set_title('Architectural Ablation and SOTA Comparison (10-Fold CV)')
    
    # Set y-axis limit to zoom in on the differences
    ax.set_ylim(1.65, 1.85)
    
    plt.savefig('ablation_comparison.png', dpi=300, bbox_inches='tight')
    print("Exported: ablation_comparison.png")

if __name__ == '__main__':
    generate_benchmark_figure()
    generate_ablation_figure()
