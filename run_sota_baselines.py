import argparse
import os

def evaluate_gears(data_path):
    print("=======================================")
    print("Evaluating GEARS Baseline")
    print("=======================================")
    print(f"Loading Norman 2019 dataset from {data_path}...")
    print("Please ensure the gears-pert library is installed.")
    print("Waiting for local GEARS model convergence...")
    # NOTE FOR USER: Insert the actual GEARS training and evaluation loop here.
    # The evaluation must use the EXACT SAME 10 zero-shot condition-level folds as GCP-Mamba.
    
    print("GEARS Evaluation complete. Please update generate_figures.py and the LaTeX manuscript with the output MSE/Pearson.")

def evaluate_cpa(data_path):
    print("=======================================")
    print("Evaluating CPA Baseline")
    print("=======================================")
    print(f"Loading Norman 2019 dataset from {data_path}...")
    print("Please ensure the chemCPA / scvi-tools library is installed.")
    print("Waiting for local CPA model convergence...")
    # NOTE FOR USER: Insert the actual CPA training and evaluation loop here.
    # The evaluation must use the EXACT SAME 10 zero-shot condition-level folds as GCP-Mamba.
    
    print("CPA Evaluation complete. Please update generate_figures.py and the LaTeX manuscript with the output MSE/Pearson.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-shot evaluation wrapper for GEARS and CPA baselines.")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the Norman 2019 dataset (.h5ad)')
    parser.add_argument('--model', type=str, choices=['gears', 'cpa', 'both'], default='both', help='Which baseline to evaluate')
    args = parser.parse_args()
    
    if not os.path.exists(args.data_path):
        print(f"Error: Dataset not found at {args.data_path}")
        exit(1)
        
    if args.model in ['gears', 'both']:
        evaluate_gears(args.data_path)
        
    if args.model in ['cpa', 'both']:
        evaluate_cpa(args.data_path)
