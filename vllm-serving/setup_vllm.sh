#!/usr/bin/env bash
# ==============================================================================
# vLLM Setup — creates a venv and installs vLLM (WSL2 / Linux)
# Run once.  After this, start.sh and start_vllm.ps1 activate the venv
# automatically.
# ==============================================================================
set -euo pipefail

VLLM_VENV="${VLLM_VENV:-$HOME/vllm-env}"

echo "=== vLLM Setup ==="
echo "  Venv location: $VLLM_VENV"
echo ""

# 1. Create venv (skip if it already exists)
if [ ! -d "$VLLM_VENV" ]; then
    echo "Creating Python venv at $VLLM_VENV ..."
    python3 -m venv "$VLLM_VENV"
else
    echo "Venv already exists at $VLLM_VENV — reusing."
fi

# 2. Activate
# shellcheck source=/dev/null
source "$VLLM_VENV/bin/activate"

# 3. Upgrade pip
python -m pip install --upgrade pip

# 4. Install vLLM (includes torch with CUDA)
echo ""
echo "Installing vLLM (this may take several minutes)..."
pip install vllm --upgrade

# 5. Upgrade transformers & huggingface_hub for Gemma 4 support
# vLLM's pip metadata pins transformers<5, but Gemma 4 (gemma4 arch)
# requires transformers >=5.5.0.  The override is safe — vLLM 0.19.0
# imports correctly with transformers 5.x.
echo ""
echo "Upgrading transformers & huggingface_hub for Gemma 4 support..."
pip install 'transformers>=5.5.0' --no-deps
pip install --upgrade huggingface_hub

# 6. Verify
echo ""
echo "=== Verification ==="
python -c "import vllm; print(f'vLLM {vllm.__version__} installed successfully')"
python -c "import torch; print(f'PyTorch {torch.__version__}  CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'transformers {transformers.__version__}')"

echo ""
echo "=== Done ==="
echo "Venv:  $VLLM_VENV"
echo "Start: ./start.sh"
echo "  or:  .\\start_vllm.ps1  (from Windows)"
