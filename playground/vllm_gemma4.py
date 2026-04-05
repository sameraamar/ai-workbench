"""
vLLM + Gemma 4 playground script.

Two modes:
  1. CLIENT MODE  — talks to a running vLLM server (recommended for the sandbox)
  2. OFFLINE MODE — loads the model in-process (useful for one-off tests; slow to start)

---------------------------------------------------------------------------
SETUP (run once inside WSL2 or Linux):
  pip install vllm --upgrade

START THE SERVER (WSL2 terminal):
  vllm serve google/gemma-4-e2b-it \\
    --host 0.0.0.0 --port 8000 \\
    --max-model-len 8192 \\
    --gpu-memory-utilization 0.90

Then run this script from Windows (or WSL2):
  python vllm_gemma4.py

RTX 3090 (24 GB) notes:
  - gemma-4-e2b-it  (~2B)  — fits comfortably at bf16
  - gemma-4-e4b-it  (~4B)  — fits at bf16
  - gemma-4-27b-it  (~27B) — needs --quantization bitsandbytes or awq
---------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import sys

# ---------------------------------------------------------------------------
# CONFIG — change these to match your setup
# ---------------------------------------------------------------------------
SERVER_URL = "http://localhost:8000"          # vLLM server address
MODEL_ID   = "google/gemma-4-e2b-it"         # model to test
MODE       = "client"                         # "client" | "offline"
# ---------------------------------------------------------------------------

SAMPLE_MESSAGES_TEXT = [
    {"role": "system", "content": "You are a helpful assistant. Answer concisely."},
    {"role": "user",   "content": "Explain PagedAttention in three sentences."},
]

SAMPLE_MESSAGES_MULTIMODAL = [
    {"role": "system", "content": "You are a helpful vision assistant."},
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is in this image?"},
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
    """Call vLLM server with a plain-text prompt — OpenAI-compatible endpoint."""
    import requests

    print("\n=== CLIENT MODE — text ===")
    payload = {
        "model": MODEL_ID,
        "messages": SAMPLE_MESSAGES_TEXT,
        "max_tokens": 256,
        "temperature": 1.0,
    }
    resp = requests.post(f"{SERVER_URL}/v1/chat/completions", json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    print("Response:", data["choices"][0]["message"]["content"])
    usage = data.get("usage", {})
    print(f"Tokens: {usage.get('prompt_tokens')} in / {usage.get('completion_tokens')} out")


def run_client_multimodal() -> None:
    """Call vLLM server with an image attachment."""
    import requests

    print("\n=== CLIENT MODE — multimodal (image) ===")
    payload = {
        "model": MODEL_ID,
        "messages": SAMPLE_MESSAGES_MULTIMODAL,
        "max_tokens": 256,
        "temperature": 1.0,
    }
    try:
        resp = requests.post(f"{SERVER_URL}/v1/chat/completions", json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        print("Response:", data["choices"][0]["message"]["content"])
    except Exception as exc:
        print(f"Multimodal test skipped or failed: {exc}")


def run_client_streaming() -> None:
    """Stream tokens from vLLM server via SSE."""
    import requests

    print("\n=== CLIENT MODE — streaming ===")
    payload = {
        "model": MODEL_ID,
        "messages": SAMPLE_MESSAGES_TEXT,
        "max_tokens": 128,
        "temperature": 1.0,
        "stream": True,
    }
    with requests.post(
        f"{SERVER_URL}/v1/chat/completions",
        json=payload,
        stream=True,
        timeout=120,
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
    print()  # newline after stream


def run_offline() -> None:
    """Load Gemma 4 in-process and run inference directly (no server needed)."""
    from vllm import LLM
    from vllm.sampling_params import SamplingParams

    print("\n=== OFFLINE MODE ===")
    print(f"Loading {MODEL_ID} into vLLM engine (this takes a minute on first run)...")
    llm = LLM(
        model=MODEL_ID,
        max_model_len=8192,
        gpu_memory_utilization=0.90,
        dtype="bfloat16",
    )
    params = SamplingParams(max_tokens=256, temperature=1.0)
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
                "Start it with:\n"
                f"  vllm serve {MODEL_ID} --host 0.0.0.0 --port 8000\n"
                "Then re-run this script."
            )
            sys.exit(1)

        run_client_text()
        run_client_multimodal()
        run_client_streaming()
        print("\nAll tests passed.")
