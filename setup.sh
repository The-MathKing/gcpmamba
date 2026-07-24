#!/bin/bash
# GCP-Mamba Dependency Installation Script

echo "Setting up GCP-Mamba Environment..."

# Update package lists
sudo apt-get update && sudo apt-get install -y build-essential

# Ensure CUDA 12.x compiler (nvcc) is in PATH
export PATH=/usr/local/cuda-12/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12/lib64:$LD_LIBRARY_PATH

# 1. Install PyTorch (Stable for CUDA 12.1 or 12.4 depending on specific driver)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2. Install PyTorch Geometric (PyG) and its dependencies
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html

# 3. Install core Mamba dependencies
pip install causal-conv1d>=1.4.0
pip install mamba-ssm>=2.0.0

# 4. Install additional data processing and evaluation tools
pip install scanpy pandas numpy scipy scikit-learn networkx

echo "Environment setup complete."
