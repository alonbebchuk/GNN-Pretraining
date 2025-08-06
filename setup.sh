#!/bin/bash
# Simple setup script

set -e  # Exit on any error

echo "🚀 Setting up GNN Research Environment..."

# Check CUDA version
echo "🔍 Checking CUDA version..."
nvidia-smi
nvcc --version || echo "NVCC not found"

# Install PyTorch with CUDA support
echo "📦 Installing PyTorch with CUDA support..."
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric
echo "📦 Installing PyTorch Geometric..."
pip install torch-geometric==2.4.0
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0%2Bcu118.html

# Install other dependencies
echo "📦 Installing other dependencies..."
pip install -r requirements.txt

# Quick validation
echo "✅ Validating installation..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')

import torch_geometric
print(f'PyTorch Geometric: {torch_geometric.__version__}')

# Test basic functionality
x = torch.randn(100, 10).cuda() if torch.cuda.is_available() else torch.randn(100, 10)
print(f'Test tensor on: {x.device}')
print('✅ Basic setup working!')
"

echo "🎉 Setup complete!"
echo "✅ Ready for experiments!"