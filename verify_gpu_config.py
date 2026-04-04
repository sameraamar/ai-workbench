#!/usr/bin/env python3
"""
Environment Variable Verification Script

This script checks that all GPU-related environment variables are properly
configured and shows their current values and effects.

Usage:
    python verify_gpu_config.py
    
    # Or with specific settings:
    GEMMA_FORCE_CPU=1 python verify_gpu_config.py
    GEMMA_GPU_ID=1 python verify_gpu_config.py
"""

import os
import sys
from pathlib import Path

# Add model-serving src to path for imports
model_serving_src = Path(__file__).parent / "model-serving" / "src"
if model_serving_src.exists():
    sys.path.insert(0, str(model_serving_src))

def check_environment_variables():
    """Check all GPU-related environment variables and their values."""
    print("🔧 GPU Configuration Verification")
    print("=" * 50)
    
    # Check environment variables
    env_vars = {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "Not set"),
        "GEMMA_FORCE_CPU": os.environ.get("GEMMA_FORCE_CPU", "0"),  
        "GEMMA_GPU_ID": os.environ.get("GEMMA_GPU_ID", "Not set"),
        "GEMMA_DEVICE_MAP": os.environ.get("GEMMA_DEVICE_MAP", "auto"),
        "GEMMA_QUANTIZE_4BIT": os.environ.get("GEMMA_QUANTIZE_4BIT", "0"),
    }
    
    print("Environment Variables:")
    for var, value in env_vars.items():
        print(f"  {var:25} = {value}")
    print()
    
    return env_vars

def check_pytorch_availability():
    """Check PyTorch CUDA availability and GPU detection."""
    try:
        import torch
        print("PyTorch GPU Status:")
        print(f"  CUDA Available:        {torch.cuda.is_available()}")
        print(f"  GPU Count:             {torch.cuda.device_count()}")
        
        if torch.cuda.is_available():
            print(f"  Current Device:        {torch.cuda.current_device()}")
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                memory_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"  GPU {i}:                 {name} ({memory_gb:.1f} GB)")
        else:
            print("  No CUDA GPUs available or PyTorch not installed with CUDA support")
        print()
        
    except ImportError:
        print("⚠️  PyTorch not available - install with: pip install torch")
        print()

def check_config_parsing():
    """Check how the configuration would be parsed."""
    try:
        from gemma_serving.config import ServingConfig
        
        print("Parsed Configuration:")
        config = ServingConfig()
        print(f"  force_cpu:             {config.force_cpu}")
        print(f"  gpu_id:                {config.gpu_id}")
        print(f"  device_map:            {config.device_map}")
        print(f"  quantize_4bit:         {config.quantize_4bit}")
        print()
        
    except ImportError:
        print("⚠️  Could not import gemma_serving.config")
        print("    Make sure you're running from the repository root")
        print()

def main():
    """Run all verification checks."""
    env_vars = check_environment_variables()
    check_pytorch_availability()
    check_config_parsing()
    
    # Provide recommendations
    print("🎯 Recommendations:")
    
    cuda_visible = env_vars.get("CUDA_VISIBLE_DEVICES", "Not set")
    force_cpu = env_vars.get("GEMMA_FORCE_CPU", "0")
    
    if force_cpu in ("1", "true", "True"):
        print("  • CPU mode is forced - AI inference will be slow")
        print("    Remove GEMMA_FORCE_CPU to enable GPU")
        
    elif cuda_visible == "":
        print("  • CUDA_VISIBLE_DEVICES is empty - no GPUs visible")
        print("    Set CUDA_VISIBLE_DEVICES=0 to use first GPU")
        
    elif cuda_visible == "Not set":
        print("  • Using automatic GPU detection (recommended)")
        
    else:
        print(f"  • Using GPU(s): {cuda_visible}")
    
    print()
    print("📚 For more configuration options, see:")
    print("    docs/gpu-selection-guide.md")

if __name__ == "__main__":
    main()