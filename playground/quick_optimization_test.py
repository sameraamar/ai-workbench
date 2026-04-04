#!/usr/bin/env python3
"""
Quick Optimized Performance Test

This script quickly tests the optimized Gemma 4 performance with all optimizations enabled.
It provides an immediate comparison to see the performance improvement on your RTX 3090.

Usage:
    python quick_optimization_test.py
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

from gemma_serving.config import ServingConfig, GenerationSettings
from gemma_serving.gemma_service import GemmaService

def setup_logging():
    """Setup logging to see optimization details."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def test_optimized_performance():
    """Test optimized configuration and show results."""
    print("=" * 60)
    print("🚀 OPTIMIZED GEMMA 4 PERFORMANCE TEST")
    print("=" * 60)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("=" * 60)
    
    # Test with optimizations enabled
    optimized_config = ServingConfig(
        model_id="google/gemma-4-E2B-it",
        quantize_4bit=True,  # 4-bit quantization for speed
        enable_torch_compile=True,  # Compile for faster inference
        enable_flash_attention=True,  # Flash Attention 2
        enable_memory_optimizations=True,  # Memory optimizations
        optimize_for_inference=True,  # General inference optimizations
        torch_compile_mode="default"  # Use default compile mode
    )
    
    print(f"🔧 Loading optimized model (E2B with all optimizations)...")
    print(f"   • 4-bit quantization: ✅")
    print(f"   • torch.compile(): ✅")
    print(f"   • Flash Attention: ✅")
    print(f"   • Memory optimizations: ✅")
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Initialize service
    service = GemmaService(optimized_config)
    
    # Load model and measure time
    load_start = time.perf_counter()
    service.ensure_loaded()
    load_time = time.perf_counter() - load_start
    
    # Get memory usage
    if torch.cuda.is_available():
        memory_used = torch.cuda.memory_allocated(0) / 1024**3  # Convert to GB
        print(f"✅ Model loaded in {load_time:.2f}s, using {memory_used:.2f} GB GPU memory")
    else:
        print(f"✅ Model loaded in {load_time:.2f}s (CPU mode)")
    
    # Test generation
    test_prompt = "Write a compelling product description for a vintage leather jacket that would work well on eBay."
    messages = [{"role": "user", "content": test_prompt}]
    generation_settings = GenerationSettings(
        max_new_tokens=150,
        temperature=0.7,
        stream_output=False
    )
    
    print(f"\n🧪 Running performance test...")
    print(f"   Prompt: {test_prompt[:60]}...")
    
    # Warmup run
    print("   Warming up GPU...")
    service.generate(messages, generation_settings)
    
    # Timed runs for average
    times = []
    token_rates = []
    
    for i in range(3):
        print(f"   Run {i+1}/3...")
        start_time = time.perf_counter()
        result = service.generate(messages, generation_settings)
        end_time = time.perf_counter()
        
        elapsed = end_time - start_time
        times.append(elapsed)
        
        # Calculate tokens per second
        output_tokens = result.get('output_token_count', 0)
        if output_tokens and elapsed > 0:
            tokens_per_sec = output_tokens / elapsed
            token_rates.append(tokens_per_sec)
            print(f"      Time: {elapsed:.2f}s, Tokens: {output_tokens}, Rate: {tokens_per_sec:.2f} tok/s")
    
    # Calculate and display results
    avg_time = sum(times) / len(times)
    avg_rate = sum(token_rates) / len(token_rates) if token_rates else 0
    
    print(f"\n" + "=" * 60)
    print(f"📊 RESULTS:")
    print(f"=" * 60)
    print(f"Average Generation Time: {avg_time:.2f} seconds")
    print(f"Average Token Rate: {avg_rate:.2f} tokens/second")
    print(f"Model Load Time: {load_time:.2f} seconds")
    print(f"GPU Memory Usage: {memory_used:.2f} GB")
    
    # Show sample output
    if 'text' in result:
        print(f"\n📝 Sample Output (last run):")
        print(f"-" * 40)
        print(result['text'][:200] + ("..." if len(result['text']) > 200 else ""))
    
    # Performance comparison
    print(f"\n🏁 PERFORMANCE ANALYSIS:")
    print(f"-" * 40)
    if avg_rate > 10:
        print(f"🚀 Excellent performance! ({avg_rate:.2f} tok/s)")
    elif avg_rate > 7:
        print(f"✅ Good performance ({avg_rate:.2f} tok/s)")
    elif avg_rate > 5:
        print(f"⚡ Decent performance ({avg_rate:.2f} tok/s)")
    else:
        print(f"🐌 Slow performance ({avg_rate:.2f} tok/s) - check CUDA installation")
    
    print(f"\n💡 COMPARISON TO BASELINE:")
    baseline_rate = 7.4  # Your previous measured performance
    if avg_rate > baseline_rate:
        improvement = ((avg_rate - baseline_rate) / baseline_rate) * 100
        print(f"🎉 {improvement:.1f}% faster than baseline ({baseline_rate} tok/s)!")
    else:
        decline = ((baseline_rate - avg_rate) / baseline_rate) * 100
        print(f"⚠️  {decline:.1f}% slower than baseline - optimizations may need tuning")
    
    # Environment variable suggestions
    print(f"\n🔧 TO USE THESE OPTIMIZATIONS PERMANENTLY:")
    print(f"   Add to your environment (.env file or shell):")
    print(f"   export GEMMA_QUANTIZE_4BIT=1")
    print(f"   export GEMMA_TORCH_COMPILE=1")
    print(f"   export GEMMA_FLASH_ATTENTION=1")
    print(f"   export GEMMA_MEMORY_OPT=1")
    print(f"   export GEMMA_INFERENCE_OPT=1")
    
    return avg_rate

def main():
    setup_logging()
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Install CUDA-enabled PyTorch for GPU acceleration.")
        print("   Run: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        return
    
    try:
        final_rate = test_optimized_performance()
        print(f"\n✅ Test completed successfully! Final rate: {final_rate:.2f} tokens/second")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()