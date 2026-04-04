#!/usr/bin/env python3
"""
Start the Gemma serving server with proper .env loading and Python path
"""
import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv

def main():
    # Ensure we're in the model-serving directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print("🚀 Starting Gemma 4 Model Server")
    print("=" * 50)
    
    # Load .env file
    env_file = Path(".env")
    if env_file.exists():
        load_dotenv()
        print("✅ Loaded .env configuration")
    else:
        print("⚠️  No .env file found - using defaults")
    
    # Verify key settings
    quantize = os.getenv("GEMMA_QUANTIZE_4BIT", "0")
    compile_mode = os.getenv("GEMMA_TORCH_COMPILE", "1") 
    memory_opt = os.getenv("GEMMA_MEMORY_OPT", "1")
    
    print(f"   Quantization: {'DISABLED' if quantize == '0' else 'ENABLED'}")
    print(f"   Torch Compile: {'DISABLED' if compile_mode == '0' else 'ENABLED'}")
    print(f"   Memory Opt: {'ENABLED' if memory_opt == '1' else 'DISABLED'}")
    
    # Set Python path to include src/
    env = os.environ.copy()
    src_path = str(script_dir / "src")
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{src_path}:{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = src_path
    
    print(f"   Python path: {src_path}")
    print("")
    print("🌐 Starting FastAPI server on http://127.0.0.1:8000")
    print("   Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Start uvicorn server
    cmd = [
        sys.executable, "-m", "uvicorn", 
        "gemma_serving.app:app",
        "--host", "127.0.0.1",
        "--port", "8000", 
        "--reload"
    ]
    
    try:
        subprocess.run(cmd, env=env, cwd=script_dir)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server failed to start: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())