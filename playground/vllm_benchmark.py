"""
vLLM + Gemma 4 benchmark: text throughput, multimodal (image via file:// URI).

Measures tokens/second, latency, and streaming TTFT for the vLLM backend.

Run with vLLM server already up:
    python playground/vllm_benchmark.py

Results are printed as a table and appended to playground/results.json.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SERVER_URL = "http://localhost:8000"
MODEL_ID = "google/gemma-4-E2B-it"
SHARED_MEDIA_DIR_WIN = Path(r"C:\ai-workbench\shared-media")
SHARED_MEDIA_DIR_WSL = "/mnt/c/ai-workbench/shared-media"
TEST_IMAGE_PATH = Path(__file__).parent.parent / "docs" / "screenshots" / "test-image.png"

RUNS = 3  # repetitions per scenario


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def post_completion(messages: list, *, stream: bool = False, max_tokens: int = 200) -> dict:
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "stream": stream,
        "stream_options": {"include_usage": True} if stream else None,
    }
    if not stream:
        del payload["stream_options"]
    resp = requests.post(
        f"{SERVER_URL}/v1/chat/completions",
        json=payload,
        stream=stream,
        timeout=300,
    )
    resp.raise_for_status()
    if not stream:
        return resp.json()

    # Stream: collect chunks
    first_token_time = None
    text = ""
    usage = {}
    start = time.perf_counter()
    for line in resp.iter_lines():
        if not line or line == b"data: [DONE]":
            continue
        raw = line.removeprefix(b"data: ")
        try:
            chunk = json.loads(raw)
        except Exception:
            continue
        if first_token_time is None and chunk.get("choices"):
            delta = chunk["choices"][0].get("delta", {})
            if delta.get("content"):
                first_token_time = time.perf_counter() - start
        for c in chunk.get("choices", []):
            text += c.get("delta", {}).get("content", "") or ""
        if "usage" in chunk:
            usage = chunk["usage"]
    return {"text": text, "usage": usage, "ttft": first_token_time}


def copy_image_to_shared(src: Path) -> str | None:
    """Copy test image to shared media dir and return its WSL file:// URI."""
    if not SHARED_MEDIA_DIR_WIN.exists():
        return None
    if not src.exists():
        return None
    dest = SHARED_MEDIA_DIR_WIN / src.name
    dest.write_bytes(src.read_bytes())
    return f"file://{SHARED_MEDIA_DIR_WSL}/{src.name}"


def run_scenario(name: str, messages: list, *, stream: bool = True, max_tokens: int = 200) -> dict:
    latencies = []
    ttfts = []
    prompt_toks = []
    completion_toks = []

    print(f"\n  {name} (×{RUNS})...")
    for i in range(RUNS):
        t0 = time.perf_counter()
        result = post_completion(messages, stream=stream, max_tokens=max_tokens)
        elapsed = time.perf_counter() - t0

        usage = result.get("usage", {})
        pt = usage.get("prompt_tokens", 0)
        ct = usage.get("completion_tokens", 0)
        ttft = result.get("ttft")

        latencies.append(elapsed)
        prompt_toks.append(pt)
        completion_toks.append(ct)
        if ttft is not None:
            ttfts.append(ttft)

        tps = ct / elapsed if elapsed > 0 else 0
        print(f"    [{i+1}/{RUNS}] {elapsed:.2f}s | {ct} tokens | {tps:.1f} tok/s"
              + (f" | TTFT {ttft*1000:.0f}ms" if ttft else ""))

    avg_lat   = statistics.mean(latencies)
    avg_ct    = statistics.mean(completion_toks)
    avg_tps   = avg_ct / avg_lat if avg_lat > 0 else 0
    avg_ttft  = statistics.mean(ttfts) * 1000 if ttfts else None  # ms

    return {
        "scenario": name,
        "runs": RUNS,
        "avg_latency_s": round(avg_lat, 3),
        "avg_completion_tokens": round(avg_ct, 1),
        "avg_tokens_per_second": round(avg_tps, 2),
        "avg_ttft_ms": round(avg_ttft, 1) if avg_ttft else None,
        "all_latencies_s": [round(x, 3) for x in latencies],
        "all_completion_tokens": completion_toks,
    }


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def scenario_short_text():
    return run_scenario(
        "Text – short (≤50 tok)",
        [{"role": "user", "content": "What is the capital of France? One word."}],
        max_tokens=20,
    )


def scenario_medium_text():
    return run_scenario(
        "Text – medium (≤200 tok)",
        [{"role": "user", "content": "Explain how transformers work in machine learning. Be concise."}],
        max_tokens=200,
    )


def scenario_long_text():
    return run_scenario(
        "Text – long (≤500 tok)",
        [{"role": "user", "content": (
            "Write a detailed comparison of bfloat16 vs float16 for deep learning inference, "
            "covering numerical range, precision, hardware support, and practical tradeoffs."
        )}],
        max_tokens=500,
    )


def scenario_image():
    uri = copy_image_to_shared(TEST_IMAGE_PATH)
    if not uri:
        print("  ⚠ Image scenario skipped — shared media dir not found or test image missing")
        return None

    print(f"  Image URI: {uri}")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image? Describe it."},
                {"type": "image_url", "image_url": {"url": uri}},
            ],
        }
    ]
    return run_scenario("Image – file:// URI (shared media)", messages, max_tokens=256)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def check_server() -> bool:
    try:
        r = requests.get(f"{SERVER_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def main():
    print("=" * 60)
    print("  vLLM Gemma 4 Benchmark")
    print("=" * 60)

    if not check_server():
        print(f"❌ vLLM not reachable at {SERVER_URL}")
        sys.exit(1)

    # Confirm model
    try:
        models = requests.get(f"{SERVER_URL}/v1/models", timeout=5).json()
        active = models["data"][0]["id"] if models.get("data") else "unknown"
        print(f"  Model:  {active}")
    except Exception:
        active = MODEL_ID

    print(f"  Server: {SERVER_URL}")
    print(f"  Runs per scenario: {RUNS}")
    print()

    results_list = []
    for fn in [scenario_short_text, scenario_medium_text, scenario_long_text, scenario_image]:
        r = fn()
        if r:
            results_list.append(r)

    # Summary table
    print("\n" + "=" * 60)
    print(f"  {'Scenario':<42} {'tok/s':>6}  {'TTFT':>6}  {'lat':>6}")
    print("-" * 60)
    for r in results_list:
        ttft_str = f"{r['avg_ttft_ms']:.0f}ms" if r.get("avg_ttft_ms") else "   n/a"
        print(f"  {r['scenario']:<42} {r['avg_tokens_per_second']:>6.1f}  {ttft_str:>6}  {r['avg_latency_s']:>5.2f}s")
    print("=" * 60)

    # Persist results
    results_file = Path(__file__).parent / "results.json"
    try:
        existing = json.loads(results_file.read_text()) if results_file.exists() else []
    except Exception:
        existing = []

    run_record = {
        "timestamp": datetime.now().isoformat(),
        "backend": "vllm",
        "model": active,
        "server_url": SERVER_URL,
        "scenarios": results_list,
    }
    existing.append(run_record)
    results_file.write_text(json.dumps(existing, indent=2))
    print(f"\n✓ Results saved → {results_file}")


if __name__ == "__main__":
    main()
