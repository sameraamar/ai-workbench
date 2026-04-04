#!/usr/bin/env python3

import requests
import time
import json

def benchmark_api():
    """Simple API benchmark test"""
    
    print("=== MODEL SERVING API BENCHMARK ===")
    
    # Check server health first
    try:
        health = requests.get("http://127.0.0.1:8000/health", timeout=5)
        print(f"Server Status: {health.json()}")
    except Exception as e:
        print(f"❌ Server not responding: {e}")
        return
    
    # Test data
    test_prompt = "Explain artificial intelligence in simple terms."
    
    test_data = {
        "messages": [
            {"role": "user", "content": test_prompt}
        ],
        "max_length": 100
    }
    
    print(f"\nPrompt: {test_prompt}")
    print("Loading model and generating response...")
    
    # Time the request
    start_time = time.time()
    
    try:
        response = requests.post(
            "http://127.0.0.1:8000/generate",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=300  # 5 minutes for model loading + generation
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ SUCCESS!")
            print(f"Response: {result.get('response', 'No response field')}")
            print(f"\n=== PERFORMANCE METRICS ===")
            print(f"Total Time: {duration:.2f} seconds")
            
            # Check for performance stats in response
            if 'stats' in result:
                stats = result['stats']
                print(f"Tokens per Second: {stats.get('tokens_per_second', 'N/A')}")
                print(f"Total Tokens: {stats.get('total_tokens', 'N/A')}")
            
            # Check server health after generation
            health = requests.get("http://127.0.0.1:8000/health", timeout=5)
            health_data = health.json()
            print(f"Model Loaded: {health_data.get('model_loaded', False)}")
            
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out - model may still be loading")
    except Exception as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    benchmark_api()