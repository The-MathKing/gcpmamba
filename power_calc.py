import statsmodels.stats.power as smp
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=47, help="Total number of pooled conditions")
    parser.add_argument("--pooled_std", type=float, default=0.27, help="Pooled standard deviation")
    args = parser.parse_args()

    analysis_paired = smp.TTestPower()
    
    # Calculate Minimum Detectable Effect (effect size) given n, alpha, power
    # alpha is significance level
    n = args.n_samples
    alpha = 0.05
    power = 0.80
    
    effect_size = analysis_paired.solve_power(nobs=n, power=power, alpha=alpha)
    
    mde_raw = effect_size * args.pooled_std
    
    print("=== Post-Hoc Power Analysis ===")
    print(f"Number of samples (N): {n}")
    print(f"Alpha: {alpha}")
    print(f"Power: {power}")
    print(f"Pooled Std Dev: {args.pooled_std}")
    print(f"Minimum Detectable Effect Size (Cohen's d): {effect_size:.3f}")
    print(f"Minimum Detectable Raw Difference (Delta r): {mde_raw:.3f}")
    
if __name__ == '__main__':
    main()
