#!/usr/bin/env python3
"""
Simple Performance Comparison Test

This script tests just memory optimizations without quantization to see if 
that provides better performance on RTX 3090 which has plenty of VRAM.

Usage:
    python simple_optimization_test.py
"""

import os
import sys
import time
import torch
import logging
from pathlib import Path

# Add model-serving to Python path  
repo_root = Path(__file__).parent.parent
model_serving_path = repo_root / "model-serving" / "src"
sys.path.insert(0, str(model_serving_path))

from model_serving.config import ServingConfig, GenerationSettings
from model_serving.model_service import ModelService

def setup_logging():
    """Setup logging to see optimization details."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def test_config(name: str, config: ServingConfig):
    """Test a specific configuration."""
    print(f"\n🧪 Testing: {name}")
    print("-" * 50)
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    service = ModelService(config)
    
    # Load model and measure time
    load_start = time.perf_counter()
    service.ensure_loaded()
    load_time = time.perf_counter() - load_start
    
    # Get memory usage
    memory_used = torch.cuda.memory_allocated(0) / 1024**3 if torch.cuda.is_available() else 0
    print(f"✅ Load time: {load_time:.1f}s, Memory: {memory_used:.1f}GB")
    
    # Quick test
    test_prompt = "Write a short, compelling eBay description for a vintage leather jacket."
    messages = [{"role": "user", "content": test_prompt}]
    settings = GenerationSettings(max_new_tokens=100, temperature=0.7, stream_output=False)
    
    # Warmup + 2 timed runs
    service.generate(messages, settings)  # warmup
    
    times = []
    rates = []
    for i in range(2):
        start = time.perf_counter()
        result = service.generate(messages, settings)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        
        tokens = result.get('output_token_count', 0)
        if tokens and elapsed > 0:
            rate = tokens / elapsed
            rates.append(rate)
    
    avg_time = sum(times) / len(times)
    avg_rate = sum(rates) / len(rates) if rates else 0
    
    print(f"   Time: {avg_time:.2f}s")
    print(f"   Rate: {avg_rate:.2f} tok/s")
    
    return avg_rate

def main():
    setup_logging()
    
    print("=" * 60)
    print("🔍 SIMPLE PERFORMANCE COMPARISON")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    
    results = {}
    
    # Test 1: Baseline (minimal optimizations)
    results["Baseline"] = test_config("Baseline (minimal optimizations)", ServingConfig(
        model_id="google/gemma-4-E2B-it",
        quantize_4bit=False,
        enable_torch_compile=False,
        enable_flash_attention=False,
        enable_memory_optimizations=False,
        optimize_for_inference=False
    ))
    
    # Test 2: Memory optimizations only
    results["Memory Optimized"] = test_config("Memory optimizations only", ServingConfig(
        model_id="google/gemma-4-E2B-it", 
        quantize_4bit=False,
        enable_torch_compile=False,
        enable_flash_attention=False,
        enable_memory_optimizations=True,
        optimize_for_inference=True
    ))
    
    # Test 3: 4-bit quantization
    results["4-bit Quantized"] = test_config("4-bit quantization", ServingConfig(
        model_id="google/gemma-4-E2B-it",
        quantize_4bit=True,
        enable_torch_compile=False, 
        enable_flash_attention=False,
        enable_memory_optimizations=False,
        optimize_for_inference=False
    ))
    
    # Summary
    print(f"\n" + "=" * 60)
    print("📊 PERFORMANCE SUMMARY")
    print("=" * 60)
    for name, rate in results.items():
        print(f"{name:<20}: {rate:.2f} tok/s")
    
    # Find best
    best_name = max(results, key=results.get)
    best_rate = results[best_name]
    print(f"\n🏆 Best: {best_name} ({best_rate:.2f} tok/s)")
    
    # Recommendations based on results
    print(f"\n💡 RECOMMENDATIONS FOR RTX 3090:")
    if best_rate > 7.0:
        print(f"   ✅ {best_name} is fastest - use this configuration")
    else:
        print(f"   ⚠️  All configurations slower than expected")
        print(f"   ℹ️  Your previous best was 7.4 tok/s - consider that baseline")
    
        if results["Baseline"] == best_rate:
            print(f"   🎯 Use baseline configuration for best performance")
        elif results["Memory Optimized"] == best_rate:
            print(f"   🎯 Use memory optimizations: MODEL_MEMORY_OPT=1, MODEL_INFERENCE_OPT=1")
        elif results["4-bit Quantized"] == best_rate:
            print(f"   🎯 Use quantization: MODEL_QUANTIZE_4BIT=1")

if __name__ == "__main__":
    main()