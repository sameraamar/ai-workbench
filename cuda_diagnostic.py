#!/usr/bin/env python3
"""
Deep CUDA/PyTorch configuration diagnostic
"""

import torch
import os


def main():
    print("=" * 80)
    print("🔧 PYTORCH/CUDA CONFIGURATION DIAGNOSTIC")
    print("=" * 80)
    
    # PyTorch compilation settings
    print("\n🚀 PYTORCH OPTIMIZATION:")
    print(f"   torch.backends.cudnn.enabled: {torch.backends.cudnn.enabled}")
    print(f"   torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")
    print(f"   torch.backends.cudnn.deterministic: {torch.backends.cudnn.deterministic}")
    
    # CUDA settings
    print(f"\n⚡ CUDA CONFIGURATION:")
    if torch.cuda.is_available():
        print(f"   torch.version.cuda: {torch.version.cuda}")
        print(f"   torch.backends.cuda.is_built: {torch.backends.cuda.is_built()}")
        print(f"   CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
        
        # GPU capability
        capability = torch.cuda.get_device_capability(0)
        print(f"   GPU Compute Capability: {capability}")
        
        # Memory info
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   Total GPU Memory: {memory_total:.1f} GB")
        
        # CUDA optimization check
        current_stream = torch.cuda.current_stream()
        print(f"   Default CUDA Stream: {current_stream}")
    
    # Transformer optimizations
    print(f"\n🤖 TRANSFORMERS OPTIMIZATIONS:")
    try:
        from transformers import is_torch_compiled_available, is_flash_attn_available
        print(f"   Flash Attention Available: {is_flash_attn_available()}")
        print(f"   Torch Compile Available: {is_torch_compiled_available()}")
    except ImportError:
        print("   Could not check advanced optimizations")
    
    # Environment variables that affect performance
    print(f"\n🌍 PERFORMANCE ENVIRONMENT:")
    perf_vars = [
        'PYTORCH_CUDA_ALLOC_CONF',
        'CUDA_LAUNCH_BLOCKING', 
        'OMP_NUM_THREADS',
        'MKL_NUM_THREADS',
        'TOKENIZERS_PARALLELISM'
    ]
    
    for var in perf_vars:
        value = os.environ.get(var, 'Not set')
        print(f"   {var}: {value}")
    
    # Recommended optimizations
    print(f"\n💡 PERFORMANCE RECOMMENDATIONS:")
    
    if not torch.backends.cudnn.benchmark:
        print("   ⚠️  Enable cuDNN benchmark: torch.backends.cudnn.benchmark = True")
    
    if torch.backends.cudnn.deterministic:
        print("   ⚠️  Disable deterministic mode for speed: torch.backends.cudnn.deterministic = False")
    
    # Test tensor operations speed
    print(f"\n🧮 TENSOR OPERATION BENCHMARK:")
    test_tensor_performance()


def test_tensor_performance():
    """Test basic GPU tensor operation performance"""
    if not torch.cuda.is_available():
        print("   ❌ CUDA not available for testing")
        return
    
    device = torch.device('cuda:0')
    
    # Matrix multiplication test
    size = 4096
    a = torch.randn(size, size, device=device, dtype=torch.float32)
    b = torch.randn(size, size, device=device, dtype=torch.float32)
    
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    c = torch.mm(a, b)
    end.record()
    torch.cuda.synchronize()
    
    elapsed = start.elapsed_time(end)  # milliseconds
    gflops = (2 * size**3) / (elapsed * 1e6)  # GFLOPS
    
    print(f"   Matrix Multiply ({size}x{size}): {elapsed:.1f}ms")
    print(f"   Estimated GFLOPS: {gflops:.1f}")
    
    # Memory bandwidth test
    large_size = 100_000_000  # ~400MB
    x = torch.randn(large_size, device=device, dtype=torch.float32)
    
    start.record()
    y = x * 2.0  # Simple element-wise operation
    end.record()
    torch.cuda.synchronize()
    
    elapsed_bw = start.elapsed_time(end)
    bandwidth_gb_s = (large_size * 4 * 2) / (elapsed_bw * 1e6)  # GB/s (read + write)
    
    print(f"   Memory Bandwidth Test: {elapsed_bw:.1f}ms")
    print(f"   Estimated Bandwidth: {bandwidth_gb_s:.1f} GB/s")


if __name__ == "__main__":
    main()