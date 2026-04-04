#!/usr/bin/env python3
"""
Optimized Gemma 4 performance test with all PyTorch optimizations enabled
"""

from __future__ import annotations
import os
from time import perf_counter
import torch
from transformers import AutoModelForCausalLM, AutoProcessor


def main():
    print("=" * 80)
    print("⚡ OPTIMIZED GEMMA 4 PERFORMANCE TEST")
    print("=" * 80)
    
    # ENABLE ALL PYTORCH OPTIMIZATIONS
    print("\n🔧 Enabling PyTorch optimizations...")
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True  # 🚀 CRITICAL for consistent workloads!
    torch.backends.cudnn.deterministic = False
    
    # Set optimal thread counts
    torch.set_num_threads(8)  # Optimal for most CPUs
    
    # Environment optimizations
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Avoid warnings
    
    print("   ✅ cuDNN benchmark enabled")
    print("   ✅ Thread optimization set")
    
    # System info
    print(f"\n📊 SYSTEM INFO:")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Compute Capability: {torch.cuda.get_device_capability(0)}")
    
    # Run optimized tests
    test_configs = [
        {
            "name": "E2B-Fast-50tok",
            "model_id": "google/gemma-4-E2B-it", 
            "max_tokens": 50,
            "temp": 0.7,
            "top_p": 0.9,
            "top_k": 40
        },
        {
            "name": "E2B-Fast-128tok",
            "model_id": "google/gemma-4-E2B-it",
            "max_tokens": 128, 
            "temp": 0.7,
            "top_p": 0.9,
            "top_k": 40
        }
    ]
    
    for config in test_configs:
        print(f"\n🧪 TESTING: {config['name']}")
        print("-" * 50)
        result = run_optimized_test(config)
        
        if result:
            print(f"   🏆 PERFORMANCE SUMMARY:")
            print(f"      Load time: {result['load_time']:.2f}s")
            print(f"      Generation: {result['gen_time']:.3f}s")
            print(f"      Tokens/sec: {result['tokens_per_sec']:.1f} ⚡")
            print(f"      VRAM peak: {result['vram_peak']:.2f} GB")
        
        # Clear memory between tests
        torch.cuda.empty_cache()
        print("   🧹 Memory cleared")


def run_optimized_test(config):
    """Run optimized performance test"""
    model_id = config["model_id"]
    
    try:
        # 1. LOAD MODEL WITH OPTIMIZATIONS
        load_start = perf_counter()
        print(f"   📦 Loading {model_id}...")
        
        # Load processor
        processor = AutoProcessor.from_pretrained(model_id)
        
        # Load model with optimal settings
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,  # Optimal for RTX 3090
            device_map="auto",
            low_cpu_mem_usage=True  # Faster loading
        )
        
        device = next(model.parameters()).device
        load_time = perf_counter() - load_start
        print(f"   ✅ Loaded on {device} in {load_time:.2f}s")
        
        # 2. WARM UP CUDA KERNELS (CRITICAL!)
        print("   🔥 Warming up CUDA kernels...")
        warmup_input = torch.ones(1, 10, dtype=torch.long, device=device)
        with torch.inference_mode():
            _ = model(warmup_input)
        torch.cuda.synchronize()
        print("   ✅ Kernels warmed up")
        
        # 3. PREPARE OPTIMIZED INPUTS
        prompt = "Rewrite this eBay listing: Apple iPhone 15 Pro Max, excellent condition, 256GB unlocked"
        messages = [{"role": "user", "content": prompt}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, return_tensors="pt").to(device)
        
        input_len = inputs["input_ids"].shape[-1]
        print(f"   📊 Input tokens: {input_len}")
        
        # 4. OPTIMIZED GENERATION
        print(f"   🚀 Generating {config['max_tokens']} tokens...")
        
        # Clear memory stats
        torch.cuda.reset_peak_memory_stats(device)
        
        gen_start = perf_counter()
        with torch.inference_mode():  # Disable gradient computation
            outputs = model.generate(
                **inputs,
                do_sample=True,
                temperature=config["temp"],
                top_p=config["top_p"], 
                top_k=config["top_k"],
                max_new_tokens=config["max_tokens"],
                pad_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,  # Enable KV-cache
                num_beams=1  # Fastest sampling
            )
        
        torch.cuda.synchronize()  # Ensure completion
        gen_time = perf_counter() - gen_start
        
        # 5. CALCULATE METRICS
        output_len = outputs[0].shape[-1] - input_len
        tokens_per_sec = output_len / gen_time if gen_time > 0 else 0
        vram_peak = torch.cuda.max_memory_allocated(device) / 1024**3
        
        print(f"   ⚡ Generation: {gen_time:.3f}s")
        print(f"   📈 Speed: {tokens_per_sec:.1f} tokens/sec")
        print(f"   📊 Output tokens: {output_len}")
        
        return {
            "load_time": load_time,
            "gen_time": gen_time,
            "tokens_per_sec": tokens_per_sec,
            "output_tokens": output_len,
            "vram_peak": vram_peak
        }
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None


if __name__ == "__main__":
    main()