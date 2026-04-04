# Install CUDA-enabled PyTorch for model serving
# This script ensures GPU acceleration works properly

Write-Host "Installing CUDA-enabled PyTorch..." -ForegroundColor Green

# Install PyTorch with CUDA 12.1 support
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121

Write-Host "Installing other model-serving dependencies..." -ForegroundColor Green

# Install remaining requirements
pip install -r requirements.txt

Write-Host "Verifying CUDA availability..." -ForegroundColor Green

# Verify CUDA is available
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}'); print(f'PyTorch Version: {torch.__version__}')"

Write-Host "Installation complete!" -ForegroundColor Green