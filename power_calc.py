import statsmodels.stats.power as smp

def main():
    # Mean and std dev estimates from user feedback and previous results
    mu1 = 0.246
    mu2 = 0.026
    # Assume std dev of the difference is roughly around 0.3 based on Pearson std dev
    # Or pool variance: std = 0.223 and 0.315 -> pooled std ~ 0.27
    pooled_std = 0.27
    # Cohen's d
    effect_size = (mu1 - mu2) / pooled_std
    
    analysis_paired = smp.TTestPower()
    
    # Calculate required n for alpha=0.05, power=0.8
    # alpha is significance level
    n_req = analysis_paired.solve_power(effect_size=effect_size, power=0.8, alpha=0.05)
    
    print(f"Effect Size: {effect_size:.3f}")
    print(f"Required N (Seeds) for 80% power at alpha=0.05: {n_req:.2f}")
    
if __name__ == '__main__':
    main()
