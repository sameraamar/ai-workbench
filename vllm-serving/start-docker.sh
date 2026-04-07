#!/usr/bin/env bash
# ==============================================================================
# Start vLLM via Docker — production-recommended for native Linux deployments.
#
# Usage:
#   bash start-docker.sh
#   MODEL_ID=google/gemma-4-E4B-it bash start-docker.sh
#
# Prerequisites:
#   - Docker with NVIDIA Container Toolkit installed
#     https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
#   - HuggingFace token in HF_TOKEN env var (or in .env.vllm)
#
# The server exposes an OpenAI-compatible API at http://localhost:<port>.
# The UI connects to it the same way as the WSL2 venv setup.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Load .env.vllm (with CRLF stripping for cross-platform) ----------------
ENV_FILE="${SCRIPT_DIR}/.env.vllm"
if [ -f "$ENV_FILE" ]; then
    _clean_env=$(mktemp)
    tr -d '\r' < "$ENV_FILE" > "$_clean_env"
    set -a
    # shellcheck source=/dev/null
    source "$_clean_env"
    set +a
    rm -f "$_clean_env"
fi

# --- Config (env vars override .env.vllm) ------------------------------------
MODEL_ID="${MODEL_ID:-google/gemma-4-E2B-it}"
VLLM_PORT="${VLLM_PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
DTYPE="${DTYPE:-bfloat16}"
QUANTIZATION="${QUANTIZATION:-none}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-{\"image\": 24, \"video\": 1}}"
# Strip surrounding single quotes preserved from .env.vllm
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT#\'}"
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT%\'}"

HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"
# Directory shared between the UI (host) and vLLM (container) for media files.
# Must be set in .env.vllm — no hardcoded default so a misconfigured
# server fails clearly rather than silently using the wrong path.
SHARED_MEDIA_DIR="${SHARED_MEDIA_DIR:-}"

# --- Auto-detect AWQ ---------------------------------------------------------
MODEL_LOWER=$(echo "$MODEL_ID" | tr '[:upper:]' '[:lower:]')
if [[ "$MODEL_LOWER" == *"awq"* ]]; then
    DTYPE="float16"
    [ "$QUANTIZATION" = "none" ] && QUANTIZATION="awq"
fi

# --- Build vllm serve arguments ----------------------------------------------
VLLM_ARGS=(
    --model "$MODEL_ID"
    --host 0.0.0.0
    --port "$VLLM_PORT"
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --dtype "$DTYPE"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --limit-mm-per-prompt "$LIMIT_MM_PER_PROMPT"
)

if [ "$QUANTIZATION" != "none" ]; then
    VLLM_ARGS+=(--quantization "$QUANTIZATION")
fi

# fp8 halves KV-cache VRAM with minimal quality loss on Ampere+ GPUs (3090, A100, H100)
if [ "$KV_CACHE_DTYPE" != "auto" ] && [ -n "$KV_CACHE_DTYPE" ]; then
    VLLM_ARGS+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
fi

# Allow vLLM to load media files from the shared folder via file:// URIs.
if [ -n "${SHARED_MEDIA_DIR:-}" ]; then
    VLLM_ARGS+=(--allowed-local-media-path "$SHARED_MEDIA_DIR")
else
    echo "WARNING: SHARED_MEDIA_DIR is not set in .env.vllm — file:// media will be rejected."
fi

# Mistral-specific flags
if [[ "$MODEL_LOWER" == *"mistral"* ]]; then
    if [[ "$MODEL_LOWER" == *"awq"* ]]; then
        VLLM_ARGS+=(--tokenizer_mode mistral)
    else
        VLLM_ARGS+=(
            --tokenizer_mode mistral
            --config_format mistral
            --load_format mistral
        )
    fi
fi

# --- Build docker run arguments ----------------------------------------------
DOCKER_ARGS=(
    run --rm
    --gpus all
    --ipc host
    -p "${VLLM_PORT}:${VLLM_PORT}"
    -v "${HF_CACHE}:/root/.cache/huggingface"
    --name vllm-server
)

# Pass HuggingFace token if set
if [ -n "${HF_TOKEN:-}" ]; then
    DOCKER_ARGS+=(-e "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}")
fi

# Multi-GPU: enable shared memory for NCCL collective ops
if [ "$TENSOR_PARALLEL_SIZE" -gt 1 ]; then
    DOCKER_ARGS+=(--shm-size 1g)
fi

# Mount shared media directory so the container can read file:// URIs
if [ -n "${SHARED_MEDIA_DIR:-}" ] && [ -d "$SHARED_MEDIA_DIR" ]; then
    DOCKER_ARGS+=(-v "${SHARED_MEDIA_DIR}:${SHARED_MEDIA_DIR}:ro")
elif [ -n "${SHARED_MEDIA_DIR:-}" ]; then
    echo "WARNING: SHARED_MEDIA_DIR '$SHARED_MEDIA_DIR' does not exist — file:// media will fail."
fi

# --- Summary -----------------------------------------------------------------
echo ""
echo "========================================================"
echo "  vLLM Model Server (Docker)"
echo "========================================================"
echo "  Image:        $VLLM_IMAGE"
echo "  Model:        $MODEL_ID"
echo "  Port:         $VLLM_PORT"
echo "  Max tokens:   $MAX_MODEL_LEN"
echo "  VRAM util:    $GPU_MEMORY_UTILIZATION"
echo "  Dtype:        $DTYPE"
echo "  Quantization: $QUANTIZATION"
echo "  KV cache:     $KV_CACHE_DTYPE"
echo "  TP size:      $TENSOR_PARALLEL_SIZE"
echo "  MM limit:     $LIMIT_MM_PER_PROMPT"
echo "  HF cache:     $HF_CACHE"
echo "  Media path:   ${SHARED_MEDIA_DIR:-not set}"
echo "========================================================"
echo "  API:    http://localhost:${VLLM_PORT}/v1/chat/completions"
echo "  Health: http://localhost:${VLLM_PORT}/health"
echo "  Models: http://localhost:${VLLM_PORT}/v1/models"
echo "========================================================"
echo ""
echo "  To stop: docker stop vllm-server"
echo ""

# --- Launch ------------------------------------------------------------------
exec docker "${DOCKER_ARGS[@]}" "$VLLM_IMAGE" "${VLLM_ARGS[@]}"
