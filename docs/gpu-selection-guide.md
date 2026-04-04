# GPU Selection Guide

This guide shows how to control GPU usage when moving between different PCs or when working with multiple GPUs.

## Environment Variables (Recommended)

The easiest way to control GPU selection across different systems:

### PowerShell Examples:
```powershell
# Use GPU 0 only (first GPU)
$env:CUDA_VISIBLE_DEVICES="0"

# Use GPU 1 only (second GPU)  
$env:CUDA_VISIBLE_DEVICES="1"

# Use multiple GPUs (model will be split across them)
$env:CUDA_VISIBLE_DEVICES="0,1"

# Force CPU-only mode (no GPU)
$env:CUDA_VISIBLE_DEVICES=""

# Force CPU via config (alternative)
$env:GEMMA_FORCE_CPU="1"

# Use specific GPU via config
$env:GEMMA_GPU_ID="1"

# Custom device mapping
$env:GEMMA_DEVICE_MAP="cpu"
```

### Bash Examples:
```bash
# Linux/macOS
export CUDA_VISIBLE_DEVICES="0"    # GPU 0 only
export CUDA_VISIBLE_DEVICES="1"    # GPU 1 only  
export CUDA_VISIBLE_DEVICES="0,1"  # Multiple GPUs
export CUDA_VISIBLE_DEVICES=""     # CPU only

# Or use Gemma-specific configs
export GEMMA_FORCE_CPU="1"
export GEMMA_GPU_ID="1" 
```

## Configuration Files (.env)

Instead of setting environment variables manually each time, you can use configuration files:

### 1. Copy the appropriate configuration template:

**For model-serving only (GPU configuration):**
```bash
cp model-serving/.env.example model-serving/.env
```

**For UI only (connection settings):**
```bash  
cp ui/.env.example ui/.env
```

### 2. The defaults work out-of-box! Edit only if you need custom settings:

**Model-serving GPU configuration:**
```bash
# model-serving/.env
GEMMA_FORCE_CPU=0          # Set to 1 to force CPU mode
GEMMA_GPU_ID=1             # Use specific GPU (0, 1, 2, etc.)
GEMMA_DEVICE_MAP=auto      # Device mapping strategy
GEMMA_QUANTIZE_4BIT=0      # Enable 4-bit quantization to save memory
```

**UI configuration (connect to model-serving):**
```bash
# ui/.env  
GEMMA_SERVING_URL=http://localhost:8000    # Backend URL
GEMMA_MODEL_ID=google/gemma-4-E2B-it       # Model selection
```

### 3. Start each component normally:
```powershell
# Start model-serving (loads .env automatically)
cd model-serving
uvicorn gemma_serving.app:app --host 127.0.0.1 --port 8000

# Start UI (loads .env automatically)  
cd ui
streamlit run app.py
```

## Starting the Server with GPU Selection

```powershell
# Start with GPU 0 only
$env:CUDA_VISIBLE_DEVICES="0"
uvicorn gemma_serving.app:app --host 127.0.0.1 --port 8000

# Start with GPU 1 only
$env:CUDA_VISIBLE_DEVICES="1" 
uvicorn gemma_serving.app:app --host 127.0.0.1 --port 8000

# Start in CPU mode
$env:GEMMA_FORCE_CPU="1"
uvicorn gemma_serving.app:app --host 127.0.0.1 --port 8000
```

## Moving to Another PC - Checklist

When deploying to a different PC, check:

1. **GPU Availability**: Run `nvidia-smi` to see available GPUs
2. **CUDA Version**: Ensure PyTorch CUDA version matches system CUDA
3. **GPU Memory**: Check if GPU has enough VRAM for your model
4. **Set Environment**: Configure `CUDA_VISIBLE_DEVICES` for that system

### Quick System Check:
```powershell
# Use the verification script to check all settings
python verify_gpu_config.py

# Or check manually:
# Check available GPUs
nvidia-smi

# Check PyTorch GPU detection  
python -c "import torch; print('GPUs:', torch.cuda.device_count()); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

# Test your specific GPU selection
$env:CUDA_VISIBLE_DEVICES="1"  # or your target GPU
python -c "import torch; print('Selected GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Multi-GPU Scenarios

### Scenario 1: Gaming PC with RTX 4090 + RTX 3080
```powershell
# Use the faster RTX 4090 (usually GPU 0)
$env:CUDA_VISIBLE_DEVICES="0"

# Use the RTX 3080 for AI, keep 4090 for gaming
$env:CUDA_VISIBLE_DEVICES="1"

# Use both GPUs (model layers split between them)
$env:CUDA_VISIBLE_DEVICES="0,1"
```

### Scenario 2: Workstation with Multiple Identical GPUs
```powershell
# Use all available GPUs
$env:CUDA_VISIBLE_DEVICES="0,1,2,3"

# Use specific GPU for dedicated inference
$env:CUDA_VISIBLE_DEVICES="2"
```

### Scenario 3: Mixed Hardware
```powershell
# Laptop with RTX 3070 - use the only GPU
$env:CUDA_VISIBLE_DEVICES="0"

# Desktop with RTX 4090 + GTX 1080 - use the better one
$env:CUDA_VISIBLE_DEVICES="0"  # Usually the RTX 4090
```

## Configuration Precedence

The system respects this priority order:

1. **CUDA_VISIBLE_DEVICES** environment variable (highest priority)
2. **GEMMA_FORCE_CPU** environment variable
3. **GEMMA_GPU_ID** environment variable  
4. **GEMMA_DEVICE_MAP** environment variable
5. **Auto detection** (lowest priority)

## Environment Variables Reference

| Variable | Purpose | Valid Values | Default | Example |
|----------|---------|--------------|---------|---------|
| `CUDA_VISIBLE_DEVICES` | Controls which GPUs PyTorch can see | `"0"`, `"1"`, `"0,1"`, `""` (empty=CPU only) | All GPUs | `"1"` |
| `GEMMA_FORCE_CPU` | Forces CPU-only mode, ignoring GPUs | `"0"`, `"1"`, `"true"`, `"false"` | `"0"` | `"1"` |
| `GEMMA_GPU_ID` | Use specific GPU by ID | `"0"`, `"1"`, `"2"`, etc. or empty | Empty (auto) | `"1"` |
| `GEMMA_DEVICE_MAP` | Device mapping strategy | `"auto"`, `"cpu"`, custom mapping | `"auto"` | `"cpu"` |
| `GEMMA_QUANTIZE_4BIT` | Enable 4-bit quantization to save VRAM | `"0"`, `"1"`, `"true"`, `"false"` | `"0"` | `"1"` |

### Variable Interactions:
- If `CUDA_VISIBLE_DEVICES=""` → Only CPU available, other GPU settings ignored
- If `GEMMA_FORCE_CPU="1"` → CPU mode forced, GPU settings ignored  
- If `GEMMA_GPU_ID="1"` → Use GPU 1 specifically (if visible)
- If `GEMMA_QUANTIZE_4BIT="1"` → Reduce memory usage ~4× with small quality loss

## Troubleshooting

### Problem: "CUDA out of memory"
```powershell
# Try 4-bit quantization to reduce memory usage
$env:GEMMA_QUANTIZE_4BIT="1"

# Or force CPU mode
$env:GEMMA_FORCE_CPU="1"
```

### Problem: Wrong GPU selected
```powershell
# Check what GPUs are visible
python -c "import torch; print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"

# Force specific GPU
$env:CUDA_VISIBLE_DEVICES="1"  # Change to your target GPU number
```

### Problem: Model too slow
```powershell
# Ensure GPU mode is active (not CPU)
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"

# Check you're not forcing CPU mode
Remove-Item Env:GEMMA_FORCE_CPU -ErrorAction SilentlyContinue
Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
```

## Example: Production Deployment Script

```powershell
# production-start.ps1
param(
    [string]$GpuId = "0",
    [switch]$ForceCpu = $false,
    [switch]$Quantize = $false
)

if ($ForceCpu) {
    $env:GEMMA_FORCE_CPU = "1"
    Write-Host "🔧 Forcing CPU mode" -ForegroundColor Yellow
} else {
    $env:CUDA_VISIBLE_DEVICES = $GpuId
    Write-Host "🔧 Using GPU: $GpuId" -ForegroundColor Green
}

if ($Quantize) {
    $env:GEMMA_QUANTIZE_4BIT = "1"
    Write-Host "🔧 Enabling 4-bit quantization" -ForegroundColor Blue
}

# Start the server
uvicorn gemma_serving.app:app --host 0.0.0.0 --port 8000
```

Usage:
```powershell
# Use GPU 1
.\production-start.ps1 -GpuId "1"

# Use CPU only with quantization
.\production-start.ps1 -ForceCpu -Quantize

# Use GPU 0 (default)
.\production-start.ps1
```