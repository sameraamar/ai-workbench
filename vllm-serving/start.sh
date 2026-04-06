#!/usr/bin/env bash
# ==============================================================================
# Start vLLM model server
# Reads configuration from .env.vllm (or environment variables).
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Activate the vLLM venv ------------------------------------------------
VLLM_VENV="${VLLM_VENV:-$HOME/vllm-env}"
if [ -f "$VLLM_VENV/bin/activate" ]; then
    echo "Activating venv: $VLLM_VENV"
    # shellcheck source=/dev/null
    source "$VLLM_VENV/bin/activate"
else
    echo "WARNING: venv not found at $VLLM_VENV — using system Python."
    echo "         Run setup_vllm.sh first to create the venv."
fi

# --- Load .env.vllm --------------------------------------------------------
# Save any command-line overrides (set via env prefix by start_vllm.ps1)
# BEFORE sourcing the .env file, so they take precedence.
_CLI_MODEL_ID="${MODEL_ID:-}"
_CLI_VLLM_PORT="${VLLM_PORT:-}"

ENV_FILE="${SCRIPT_DIR}/.env.vllm"
if [ -f "$ENV_FILE" ]; then
    echo "Loading config from $ENV_FILE"
    # The .env.vllm lives on a Windows filesystem (/mnt/c/…) and may have
    # CRLF line endings.  Create a temp copy with CRs stripped so bash
    # doesn't choke on the trailing \r in each value.
    _clean_env=$(mktemp)
    tr -d '\r' < "$ENV_FILE" > "$_clean_env"
    set -a
    # shellcheck source=/dev/null
    source "$_clean_env"
    set +a
    rm -f "$_clean_env"
fi

# Restore command-line overrides (they win over .env.vllm).
[ -n "$_CLI_MODEL_ID" ]  && MODEL_ID="$_CLI_MODEL_ID"
[ -n "$_CLI_VLLM_PORT" ] && VLLM_PORT="$_CLI_VLLM_PORT"

# --- Defaults ---------------------------------------------------------------
MODEL_ID="${MODEL_ID:-google/gemma-4-E2B-it}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
DTYPE="${DTYPE:-bfloat16}"
QUANTIZATION="${QUANTIZATION:-none}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-'{\"image\": 10}'}"
# Strip surrounding single quotes that bash source preserves from .env.vllm
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT#\'}"  
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT%\'}"

# --- Auto-detect AWQ models (must run before building ARGS) -----------------
MODEL_LOWER=$(echo "$MODEL_ID" | tr '[:upper:]' '[:lower:]')
if [[ "$MODEL_LOWER" == *"awq"* ]]; then
    # AWQ requires float16, not bfloat16
    DTYPE="float16"
    if [ "$QUANTIZATION" = "none" ]; then
        QUANTIZATION="awq"
    fi
fi

# --- Build vllm serve arguments ---------------------------------------------
ARGS=(
    serve "$MODEL_ID"
    --host "$VLLM_HOST"
    --port "$VLLM_PORT"
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --dtype "$DTYPE"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --limit-mm-per-prompt "$LIMIT_MM_PER_PROMPT"
)

# Quantization
if [ "$QUANTIZATION" != "none" ]; then
    ARGS+=(--quantization "$QUANTIZATION")
fi

# Mistral-specific flags.
# The native Mistral loader is needed for official mistralai/* checkpoints.
# AWQ community re-packs may need different handling (tokenizer compatibility
# varies — test before relying on any specific AWQ repack).
MODEL_LOWER=$(echo "$MODEL_ID" | tr '[:upper:]' '[:lower:]')
if [[ "$MODEL_LOWER" == *"mistral"* ]]; then
    if [[ "$MODEL_LOWER" == *"awq"* ]]; then
        echo "Mistral AWQ model detected — using standard HF loader + mistral tokenizer"
        ARGS+=(--tokenizer_mode mistral)
    else
        echo "Mistral model detected — adding mistral-specific loader flags"
        ARGS+=(
            --tokenizer_mode mistral
            --config_format mistral
            --load_format mistral
            --tool-call-parser mistral
            --enable-auto-tool-choice
        )
    fi
fi

# HuggingFace token
if [ -n "${HF_TOKEN:-}" ]; then
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    echo "HuggingFace token loaded"
fi

# --- Print launch summary ---------------------------------------------------
echo ""
echo "========================================================"
echo "  vLLM Model Server"
echo "========================================================"
echo "  Model:        $MODEL_ID"
echo "  Host:         $VLLM_HOST"
echo "  Port:         $VLLM_PORT"
echo "  Max tokens:   $MAX_MODEL_LEN"
echo "  VRAM util:    $GPU_MEMORY_UTILIZATION"
echo "  Dtype:        $DTYPE"
echo "  Quantization: $QUANTIZATION"
echo "  TP size:      $TENSOR_PARALLEL_SIZE"
echo "  MM limit:     $LIMIT_MM_PER_PROMPT"
echo "========================================================"
echo "  API:  http://${VLLM_HOST}:${VLLM_PORT}/v1/chat/completions"
echo "  Health: http://${VLLM_HOST}:${VLLM_PORT}/health"
echo "  Models: http://${VLLM_HOST}:${VLLM_PORT}/v1/models"
echo "========================================================"
echo ""

# --- Launch -----------------------------------------------------------------
exec vllm "${ARGS[@]}"
