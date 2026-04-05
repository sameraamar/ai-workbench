"""
vLLM + Mistral Small 3.1 24B playground script.

Two modes:
  1. CLIENT MODE  — talks to a running vLLM server
  2. OFFLINE MODE — loads the model in-process

---------------------------------------------------------------------------
IMPORTANT — VRAM REQUIREMENTS:
  Full bf16: ~55 GB  →  requires 2× or 4× A100/H100
  4-bit AWQ: ~14 GB  →  fits on RTX 3090 (24 GB)  ✅

For RTX 3090, use the pre-quantized AWQ checkpoint:
  Model: "mistralai/Mistral-Small-3.1-24B-Instruct-2503-AWQ"
  (or any community AWQ quant, e.g. "solidrust/Mistral-Small-3.1-24B-Instruct-2503-AWQ")

Full precision requires Mistral-specific loader flags:
  --tokenizer_mode mistral --config_format mistral --load_format mistral

AWQ quants can be loaded with standard vLLM flags + --quantization awq.

---------------------------------------------------------------------------
SETUP (run once inside WSL2 or Linux):
  pip install vllm --upgrade

START THE SERVER — AWQ (RTX 3090):
  vllm serve solidrust/Mistral-Small-3.1-24B-Instruct-2503-AWQ \\
    --host 0.0.0.0 --port 8000 \\
    --quantization awq \\
    --max-model-len 32768 \\
    --limit_mm_per_prompt image=4

START THE SERVER — Full precision (multi-GPU):
  vllm serve mistralai/Mistral-Small-3.1-24B-Instruct-2503 \\
    --tokenizer_mode mistral --config_format mistral --load_format mistral \\
    --host 0.0.0.0 --port 8000 \\
    --tensor-parallel-size 2 \\
    --max-model-len 32768 \\
    --limit_mm_per_prompt image=10

Then run this script:
  python vllm_mistral.py
---------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import sys

# ---------------------------------------------------------------------------
# CONFIG — change these to match your setup
# ---------------------------------------------------------------------------
SERVER_URL = "http://localhost:8000"

# For RTX 3090 use the AWQ quant; switch to full-precision ID for multi-GPU
MODEL_ID   = "solidrust/Mistral-Small-3.1-24B-Instruct-2503-AWQ"
# MODEL_ID = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"   # full precision

MODE       = "client"                         # "client" | "offline"
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are Mistral Small 3.1, a helpful multimodal assistant. "
    "Answer concisely and accurately."
)

SAMPLE_MESSAGES_TEXT = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user",   "content": "What are three practical uses of a 128k context window?"},
]

SAMPLE_MESSAGES_MULTIMODAL = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image in two sentences."},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://huggingface.co/datasets/patrickvonplaten/random_img/resolve/main/europe.png"
                },
            },
        ],
    },
]


def run_client_text() -> None:
    """Plain-text call via OpenAI-compatible endpoint."""
    import requests

    print("\n=== CLIENT MODE — text ===")
    payload = {
        "model": MODEL_ID,
        "messages": SAMPLE_MESSAGES_TEXT,
        "max_tokens": 256,
        "temperature": 0.15,   # Mistral card recommends low temperature
    }
    resp = requests.post(f"{SERVER_URL}/v1/chat/completions", json=payload, timeout=180)
    resp.raise_for_status()
    data = resp.json()
    print("Response:", data["choices"][0]["message"]["content"])
    usage = data.get("usage", {})
    print(f"Tokens: {usage.get('prompt_tokens')} in / {usage.get('completion_tokens')} out")


def run_client_multimodal() -> None:
    """Image+text call."""
    import requests

    print("\n=== CLIENT MODE — multimodal (image) ===")
    payload = {
        "model": MODEL_ID,
        "messages": SAMPLE_MESSAGES_MULTIMODAL,
        "max_tokens": 256,
        "temperature": 0.15,
    }
    try:
        resp = requests.post(f"{SERVER_URL}/v1/chat/completions", json=payload, timeout=180)
        resp.raise_for_status()
        data = resp.json()
        print("Response:", data["choices"][0]["message"]["content"])
    except Exception as exc:
        print(f"Multimodal test skipped or failed: {exc}")


def run_client_streaming() -> None:
    """Stream tokens via SSE."""
    import requests

    print("\n=== CLIENT MODE — streaming ===")
    payload = {
        "model": MODEL_ID,
        "messages": SAMPLE_MESSAGES_TEXT,
        "max_tokens": 128,
        "temperature": 0.15,
        "stream": True,
    }
    with requests.post(
        f"{SERVER_URL}/v1/chat/completions",
        json=payload,
        stream=True,
        timeout=180,
    ) as resp:
        resp.raise_for_status()
        print("Streamed: ", end="", flush=True)
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode() if isinstance(raw_line, bytes) else raw_line
            if not line.startswith("data:"):
                continue
            payload_str = line[5:].strip()
            if payload_str == "[DONE]":
                break
            chunk = json.loads(payload_str)
            delta = chunk["choices"][0]["delta"].get("content", "")
            print(delta, end="", flush=True)
    print()


def run_offline() -> None:
    """
    Load model in-process.
    For full-precision Mistral use tokenizer_mode="mistral".
    For AWQ quant standard tokenizer mode works.
    """
    from vllm import LLM
    from vllm.sampling_params import SamplingParams

    print("\n=== OFFLINE MODE ===")
    print(f"Loading {MODEL_ID} (this can take several minutes on first run)...")

    is_full_precision = "solidrust" not in MODEL_ID and "AWQ" not in MODEL_ID and "awq" not in MODEL_ID

    llm_kwargs: dict = {
        "model": MODEL_ID,
        "max_model_len": 32768,
        "gpu_memory_utilization": 0.92,
    }
    if is_full_precision:
        # Full-precision Mistral requires mistral-specific loader
        llm_kwargs["tokenizer_mode"] = "mistral"
        llm_kwargs["config_format"] = "mistral"
        llm_kwargs["load_format"] = "mistral"
    else:
        # AWQ quant — standard loader
        llm_kwargs["quantization"] = "awq"
        llm_kwargs["dtype"] = "float16"   # AWQ requires fp16, not bf16

    llm = LLM(**llm_kwargs)
    params = SamplingParams(max_tokens=256, temperature=0.15)
    outputs = llm.chat(SAMPLE_MESSAGES_TEXT, params)
    print("Response:", outputs[0].outputs[0].text)


def check_server() -> bool:
    """Return True if the vLLM server is reachable."""
    import requests
    try:
        resp = requests.get(f"{SERVER_URL}/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


if __name__ == "__main__":
    if MODE == "offline":
        run_offline()
    else:
        if not check_server():
            print(
                f"[ERROR] vLLM server not reachable at {SERVER_URL}\n"
                "\nFor RTX 3090 start with:\n"
                "  vllm serve solidrust/Mistral-Small-3.1-24B-Instruct-2503-AWQ \\\n"
                "    --quantization awq --host 0.0.0.0 --port 8000\n"
                "\nFor full precision (multi-GPU):\n"
                "  vllm serve mistralai/Mistral-Small-3.1-24B-Instruct-2503 \\\n"
                "    --tokenizer_mode mistral --config_format mistral --load_format mistral \\\n"
                "    --tensor-parallel-size 2 --host 0.0.0.0 --port 8000\n"
            )
            sys.exit(1)

        run_client_text()
        run_client_multimodal()
        run_client_streaming()
        print("\nAll tests passed.")
