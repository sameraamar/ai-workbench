#!/usr/bin/env bash
# Start the Model Serving API — bash equivalent of start_server.ps1
# Usage: ./start_server.sh [--host HOST] [--port PORT] [--no-reload]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="127.0.0.1"
PORT=8000
NO_RELOAD=0

# --- Parse args -----------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)    HOST="$2"; shift 2 ;;
        --port)    PORT="$2"; shift 2 ;;
        --no-reload) NO_RELOAD=1; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

cd "$SCRIPT_DIR"

echo "🚀 Starting Model Server"
echo "=================================================="

# --- Load .env ------------------------------------------------------------
ENV_FILE="$SCRIPT_DIR/.env"
if [[ -f "$ENV_FILE" ]]; then
    while IFS= read -r line; do
        line="${line#"${line%%[![:space:]]*}"}"   # trim leading whitespace
        [[ -z "$line" || "$line" == \#* ]] && continue
        if [[ "$line" == *=* ]]; then
            key="${line%%=*}"
            val="${line#*=}"
            val="${val%$'\r'}"   # strip Windows CRLF trailing \r
            export "$key"="$val"
        fi
    done < "$ENV_FILE"
    echo "✅ Loaded .env configuration"
else
    echo "⚠️  No .env file found - using defaults"
fi

# --- Print config summary -------------------------------------------------
quantize=$( [[ "${GEMMA_QUANTIZE_4BIT:-0}" == "1" ]] && echo "ENABLED"  || echo "DISABLED" )
compile=$(  [[ "${GEMMA_TORCH_COMPILE:-1}"  == "0" ]] && echo "DISABLED" || echo "ENABLED"  )
memopt=$(   [[ "${GEMMA_MEMORY_OPT:-1}"     == "0" ]] && echo "DISABLED" || echo "ENABLED"  )

echo "   Quantization : $quantize"
echo "   Torch Compile: $compile"
echo "   Memory Opt   : $memopt"

# --- Set PYTHONPATH -------------------------------------------------------
SRC_PATH="$SCRIPT_DIR/src"
export PYTHONPATH="${SRC_PATH}${PYTHONPATH:+:$PYTHONPATH}"
echo "   Python path  : $SRC_PATH"

echo ""
echo "🌐 Starting FastAPI server on http://${HOST}:${PORT}"
echo "   Press Ctrl+C to stop the server"
echo "=================================================="

# --- Launch uvicorn -------------------------------------------------------
ARGS=(-m uvicorn model_serving.app:app --host "$HOST" --port "$PORT")
if [[ $NO_RELOAD -eq 0 ]]; then
    ARGS+=(--reload --reload-dir "$SRC_PATH")
fi

exec python "${ARGS[@]}"
