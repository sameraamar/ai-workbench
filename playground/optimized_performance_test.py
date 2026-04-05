#!/usr/bin/env python3
"""
Optimized Performance Test for RTX 3090

This script tests various optimization combinations to find the fastest configuration
for your RTX 3090. It will test:
1. Baseline (no optimizations)
2. 4-bit quantization only  
3. All optimizations enabled
4. Individual optimization techniques

Usage:
    python optimized_performance_test.py
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
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def print_system_info():
    """Print system information."""
    print("=" * 60)
    print("🖥️  SYSTEM INFORMATION")
    print("=" * 60)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
    torch.backends.cudnn.benchmark = True
    print("=" * 60)

def test_configuration(config_name: str, config: ServingConfig, test_prompt: str = "Write a short product description for a vintage leather jacket."):
    """Test a specific configuration and return performance metrics."""
    print(f"\n🧪 Testing: {config_name}")
    print("-" * 40)
    
    try:
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Initialize service
        service = ModelService(config)
        
        # Load model (measure time)
        load_start = time.perf_counter()
        service.ensure_loaded()
        load_time = time.perf_counter() - load_start
        
        # Get memory usage after loading
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated(0) / 1024**3  # Convert to GB
        else:
            memory_used = 0
        
        # Prepare test message
        messages = [{"role": "user", "content": test_prompt}]
        generation_settings = GenerationSettings(
            max_new_tokens=100,  # Shorter for consistent comparison
            temperature=0.7,
            stream_output=False
        )
        
        # Warmup run
        print("  Warming up...")
        service.generate(messages, generation_settings)
        
        # Timed runs
        print("  Running performance test...")
        times = []
        tokens_per_sec_list = []
        
        for i in range(3):  # 3 runs for average
            start_time = time.perf_counter()
            result = service.generate(messages, generation_settings)
            end_time = time.perf_counter()
            
            elapsed = end_time - start_time
            times.append(elapsed)
            
            # Calculate tokens per second
            output_tokens = result.get('output_token_count', 0)
            if output_tokens and elapsed > 0:
                tokens_per_sec = output_tokens / elapsed
                tokens_per_sec_list.append(tokens_per_sec)
        
        # Calculate averages
        avg_time = sum(times) / len(times)
        avg_tokens_per_sec = sum(tokens_per_sec_list) / len(tokens_per_sec_list) if tokens_per_sec_list else 0
        
        # Print results
        print(f"  ✅ Load Time: {load_time:.2f}s")
        print(f"  ✅ GPU Memory: {memory_used:.2f} GB")
        print(f"  ✅ Avg Generation Time: {avg_time:.2f}s")
        print(f"  ✅ Avg Tokens/Second: {avg_tokens_per_sec:.2f}")
        
        return {
            'config_name': config_name,
            'load_time': load_time,
            'memory_used_gb': memory_used,
            'avg_generation_time': avg_time,
            'avg_tokens_per_sec': avg_tokens_per_sec,
            'success': True
        }
        
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        return {
            'config_name': config_name,
            'error': str(e),
            'success': False
        }

def create_test_configurations():
    """Create different configurations to test."""
    base_config = ServingConfig(
        model_id="google/gemma-4-E2B-it"  # Test with E2B for faster iteration
    )
    
    configurations = [
        ("Baseline (No Optimizations)", ServingConfig(
            model_id="google/gemma-4-E2B-it",
            quantize_4bit=False,
            enable_torch_compile=False,
            enable_flash_attention=False,
            enable_memory_optimizations=False,
            optimize_for_inference=False
        )),
        
        ("4-bit Quantization Only", ServingConfig(
            model_id="google/gemma-4-E2B-it",
            quantize_4bit=True,
            enable_torch_compile=False,
            enable_flash_attention=False,
            enable_memory_optimizations=False,
            optimize_for_inference=False
        )),
        
        ("Memory Optimizations Only", ServingConfig(
            model_id="google/gemma-4-E2B-it",
            quantize_4bit=False,
            enable_torch_compile=False,
            enable_flash_attention=False,
            enable_memory_optimizations=True,
            optimize_for_inference=True
        )),
        
        ("Torch Compile Only", ServingConfig(
            model_id="google/gemma-4-E2B-it",
            quantize_4bit=False,
            enable_torch_compile=True,
            enable_flash_attention=False,
            enable_memory_optimizations=False,
            optimize_for_inference=True,
            torch_compile_mode="default"
        )),
        
        ("Flash Attention Only", ServingConfig(
            model_id="google/gemma-4-E2B-it",
            quantize_4bit=False,
            enable_torch_compile=False,
            enable_flash_attention=True,
            enable_memory_optimizations=False,
            optimize_for_inference=True
        )),
        
        ("All Optimizations (Default Mode)", ServingConfig(
            model_id="google/gemma-4-E2B-it",
            quantize_4bit=True,
            enable_torch_compile=True,
            enable_flash_attention=True,
            enable_memory_optimizations=True,
            optimize_for_inference=True,
            torch_compile_mode="default"
        )),
        
        ("All Optimizations (Max Autotune)", ServingConfig(
            model_id="google/gemma-4-E2B-it",
            quantize_4bit=True,
            enable_torch_compile=True,
            enable_flash_attention=True,
            enable_memory_optimizations=True,
            optimize_for_inference=True,
            torch_compile_mode="max-autotune"
        )),
    ]
    
    return configurations

def print_performance_summary(results):
    """Print a summary comparison of all results."""
    print("\n" + "=" * 80)
    print("🏁 PERFORMANCE SUMMARY")
    print("=" * 80)
    
    successful_results = [r for r in results if r['success']]
    
    if not successful_results:
        print("❌ No successful tests to compare")
        return
    
    # Sort by tokens per second (descending)
    successful_results.sort(key=lambda x: x['avg_tokens_per_sec'], reverse=True)
    
    print(f"{'Configuration':<35} {'Load(s)':<10} {'Mem(GB)':<10} {'Gen(s)':<10} {'Tok/s':<10}")
    print("-" * 80)
    
    baseline = None
    for i, result in enumerate(successful_results):
        if "Baseline" in result['config_name']:
            baseline = result
        
        speedup = ""
        if baseline and result != baseline:
            speedup_ratio = result['avg_tokens_per_sec'] / baseline['avg_tokens_per_sec']
            speedup = f" ({speedup_ratio:.2f}x)"
            
        print(f"{result['config_name'][:34]:<35} "
              f"{result['load_time']:<10.2f} "
              f"{result['memory_used_gb']:<10.2f} "
              f"{result['avg_generation_time']:<10.2f} "
              f"{result['avg_tokens_per_sec']:<7.2f}{speedup}")
    
    # Print recommendations
    print("\n🚀 RECOMMENDATIONS:")
    best_performance = successful_results[0]
    print(f"   Fastest: {best_performance['config_name']} ({best_performance['avg_tokens_per_sec']:.2f} tokens/sec)")
    
    # Find best memory efficiency
    successful_results.sort(key=lambda x: x['memory_used_gb'])
    best_memory = successful_results[0]
    print(f"   Most Memory Efficient: {best_memory['config_name']} ({best_memory['memory_used_gb']:.2f} GB)")
    
    print(f"\n💡 For your RTX 3090 with 24GB VRAM, you can safely use any configuration.")
    print(f"   For maximum speed, use: {best_performance['config_name']}")

def main():
    """Main function to run all performance tests."""
    setup_logging()
    print_system_info()
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This script is designed for GPU testing.")
        return
    
    print(f"\n🧪 Starting Performance Optimization Tests")
    print(f"   This may take 10-15 minutes as each model needs to load...")
    
    configurations = create_test_configurations()
    results = []
    
    for config_name, config in configurations:
        result = test_configuration(config_name, config)
        results.append(result)
        
        # Print failed tests
        if not result['success']:
            print(f"   ⚠️  {config_name} failed: {result.get('error', 'Unknown error')}")
    
    print_performance_summary(results)
    
    print(f"\n✅ Performance testing complete!")
    print(f"🔧 To use the optimal settings, set these environment variables:")
    
    # Get the best performing config
    successful_results = [r for r in results if r['success']]
    if successful_results:
        best = max(successful_results, key=lambda x: x['avg_tokens_per_sec'])
        print(f"   Best configuration: {best['config_name']}")
        
        if "4-bit" in best['config_name']:
            print("   export MODEL_QUANTIZE_4BIT=1")
        if "All Optimizations" in best['config_name']:
            print("   export MODEL_TORCH_COMPILE=1")
            print("   export MODEL_FLASH_ATTENTION=1")
            print("   export MODEL_MEMORY_OPT=1")
            print("   export MODEL_INFERENCE_OPT=1")
            if "Max Autotune" in best['config_name']:
                print("   export MODEL_COMPILE_MODE=max-autotune")

if __name__ == "__main__":
    main()