#!/usr/bin/env python3
"""
Load and verify .env configuration before starting server
"""
from dotenv import load_dotenv
import os

# Load .env file from current directory
load_dotenv()

print('🔧 RTX 3090 Optimized Settings Loaded:')
print('=' * 50)
print(f'GEMMA_MODEL_ID: {os.getenv("GEMMA_MODEL_ID", "NOT SET")}')
print(f'GEMMA_QUANTIZE_4BIT: {os.getenv("GEMMA_QUANTIZE_4BIT", "NOT SET")}')  
print(f'GEMMA_TORCH_COMPILE: {os.getenv("GEMMA_TORCH_COMPILE", "NOT SET")}')
print(f'GEMMA_MEMORY_OPT: {os.getenv("GEMMA_MEMORY_OPT", "NOT SET")}')
print(f'GEMMA_INFERENCE_OPT: {os.getenv("GEMMA_INFERENCE_OPT", "NOT SET")}')
print(f'GEMMA_FLASH_ATTENTION: {os.getenv("GEMMA_FLASH_ATTENTION", "NOT SET")}')
print('')

# Verify optimal settings
errors = []
if os.getenv("GEMMA_QUANTIZE_4BIT") != "0":
    errors.append("❌ GEMMA_QUANTIZE_4BIT should be 0 for RTX 3090 max performance")
if os.getenv("GEMMA_TORCH_COMPILE") != "0": 
    errors.append("❌ GEMMA_TORCH_COMPILE should be 0 to avoid overhead")
if os.getenv("GEMMA_MEMORY_OPT") != "1":
    errors.append("❌ GEMMA_MEMORY_OPT should be 1 for optimizations")
if os.getenv("GEMMA_INFERENCE_OPT") != "1":
    errors.append("❌ GEMMA_INFERENCE_OPT should be 1 for optimizations")

if errors:
    print("⚠️  Configuration Issues:")
    for error in errors:
        print(f"   {error}")
else:
    print("✅ Configuration is optimal for RTX 3090!")
    print("   Expected performance: ~8.2+ tokens/second")
    print("   Expected VRAM usage: ~9.5GB")