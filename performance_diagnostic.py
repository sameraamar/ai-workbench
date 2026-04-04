#!/usr/bin/env python3
"""
Performance diagnostic tool for Gemma 4 GPU inference
Tests different configurations to identify bottlenecks
"""

from __future__ import annotations
import asyncio
import os
from time import perf_counter
import torch
from transformers import AutoModelForCausalLM, AutoProcessor


def main():
    print("=" * 80)
    print("🔬 GEMMA 4 PERFORMANCE DIAGNOSTIC")
    print("=" * 80)
    
    # Environment Check
    print("\n📊 SYSTEM INFO:")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
    
    # Test configurations
    configs = [
        {
            "name": "E2B-Optimized", 
            "model_id": "google/gemma-4-E2B-it",
            "max_tokens": 50,
            "dtype": torch.bfloat16,
            "device_map": "auto"
        },
        {
            "name": "E2B-Standard", 
            "model_id": "google/gemma-4-E2B-it",
            "max_tokens": 128,
            "dtype": torch.bfloat16,
            "device_map": "auto"
        }
    ]
    
    for config in configs:
        print(f"\n🧪 TESTING: {config['name']}")
        print("-" * 40)
        run_performance_test(config)
        
        # Clear GPU memory between tests
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("   🧹 GPU cache cleared")


def run_performance_test(config):
    model_id = config["model_id"]
    max_tokens = config["max_tokens"]
    
    try:
        # Load model
        load_start = perf_counter()
        print(f"   📦 Loading {model_id}...")
        
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=config["dtype"],
            device_map=config["device_map"]
        )
        
        # Get actual device
        device = next(model.parameters()).device
        load_time = perf_counter() - load_start
        print(f"   ✅ Model loaded on {device} in {load_time:.2f}s")
        
        # Memory check
        if device.type == "cuda":
            allocated = torch.cuda.memory_allocated(device) / 1024**3
            print(f"   💾 VRAM allocated: {allocated:.2f} GB")
        
        # Performance test
        prompt = "Rewrite this product description for eBay: Apple iPhone 15 Pro Max in perfect condition"
        
        # Prepare inputs
        messages = [{"role": "user", "content": prompt}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, return_tensors="pt").to(device)
        
        print(f"   🚀 Generating {max_tokens} tokens...")
        
        # Generation test
        gen_start = perf_counter()
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                max_new_tokens=max_tokens,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        gen_time = perf_counter() - gen_start
        
        # Results
        input_len = inputs["input_ids"].shape[-1] 
        output_len = outputs[0].shape[-1] - input_len
        tokens_per_sec = output_len / gen_time if gen_time > 0 else 0
        
        print(f"   ⚡ Generation: {gen_time:.3f}s")
        print(f"   📈 Tokens/sec: {tokens_per_sec:.1f}")
        print(f"   📊 Input tokens: {input_len}")
        print(f"   📊 Output tokens: {output_len}")
        
        # Memory after generation
        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(device) / 1024**3
            print(f"   🔥 Peak VRAM: {peak:.2f} GB")
        
        return {
            "load_time": load_time,
            "gen_time": gen_time, 
            "tokens_per_sec": tokens_per_sec,
            "output_tokens": output_len
        }
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None


if __name__ == "__main__":
    main()