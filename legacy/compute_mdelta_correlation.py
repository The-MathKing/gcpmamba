import numpy as np
import scipy.stats as stats
import argparse

def compute_correlation(mdelta_norms, distances):
    """
    Computes the Pearson correlation between the L2 norm of the M_delta tensor
    and the Gene Ontology shortest-path distances across all evaluated gene pairs.
    """
    # Filter out identical pairs or disconnected pairs if necessary
    valid_mask = (distances > 0) & (distances < np.inf)
    valid_mdelta = mdelta_norms[valid_mask]
    valid_distances = distances[valid_mask]
    
    # Calculate Pearson correlation
    r, p = stats.pearsonr(valid_mdelta, valid_distances)
    
    print("=======================================")
    print("M_delta - GO Topology Correlation")
    print("=======================================")
    print(f"Number of valid gene pairs: {len(valid_mdelta)}")
    print(f"Pearson r: {r:.4f}")
    print(f"p-value:   {p:.2e}")
    print("=======================================")
    
    import json
    with open('correlation_results.json', 'w') as f:
        json.dump({'pearson': float(r), 'p_value': float(p)}, f, indent=2)
    print("Saved correlation_results.json")
    
    return r, p

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Systematic analysis of M_delta conditioning tensor.")
    parser.add_argument('--mdelta_path', type=str, default='mdelta_norms.npy', help='Path to extracted M_delta norms')
    parser.add_argument('--distance_path', type=str, default='go_distances.npy', help='Path to GO distance matrix')
    args = parser.parse_args()
    
    print(f"Loading data from {args.mdelta_path} and {args.distance_path}...")
    try:
        mdelta_norms = np.load(args.mdelta_path)
        distances = np.load(args.distance_path)
        compute_correlation(mdelta_norms, distances)
    except FileNotFoundError:
        print("Error: Required .npy files not found.")
        print("Please ensure you have extracted the M_delta L2 norms and GO distances from the converged model.")
        print("Example: np.save('mdelta_norms.npy', mdelta_norms)")
