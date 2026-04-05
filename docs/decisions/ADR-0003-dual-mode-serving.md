# ADR-0003 — Dual-Mode Serving: vLLM (WSL2) + Windows-Native Transformers

**Status:** ACCEPTED  
**Date:** 2026-04-05  
**Author:** Copilot + human review

---

## Context

The vLLM migration (ADR-0002) delivers the best inference engine for production use, but it
requires WSL2 or native Linux — vLLM does not run natively on Windows.  This creates a friction
point: a developer on Windows who just wants to run the playground or iterate on the UI has to
set up WSL2 before anything works.

Two questions surfaced:

1. **Can we keep a simple "just Python on Windows" path for quick iteration?**
2. **Do we need separate repositories or branches to maintain both modes?**

---

## Decision

**Single repo, single branch, two backend modes** selected at *start time* by which script you run.

The UI's `serving_client.py` speaks the **OpenAI-compatible API** (`/v1/chat/completions`,
`/v1/models`, `/health`).  Any backend that implements this protocol works — the UI never
knows or cares which backend is running.

| Mode | Backend | Platform | Start command | Best for |
|---|---|---|---|---|
| **Mode 1: vLLM** | vLLM server (PagedAttention, continuous batching) | WSL2 / Linux | `start_vllm.ps1` or `start.sh` | Production-like perf, Mistral correctness, benchmarks |
| **Mode 2: Windows-native** | Existing FastAPI + GemmaService + OpenAI shim | Windows Python | `start_server.ps1` | Quick UI iteration, no WSL2 required |

### Why not two repos or branches?

- **Shared code:** The UI, playground scripts, docs, and tests are identical in both modes.
  Duplicating them across repos creates a maintenance nightmare.
- **The only difference is backend startup.**  Everything else — model profiles, prompts,
  conversation state, media handling — is backend-agnostic.
- **The OpenAI API contract is the boundary.**  As long as both backends serve
  `/v1/chat/completions`, the rest of the stack doesn't care.  This is the same contract
  that OpenAI, Anthropic, Ollama, llama.cpp, TGI, and dozens of other tools use.

---

## What Was Built

### 1. OpenAI-compatible shim (`openai_compat.py`)

New file: `model-serving/src/gemma_serving/openai_compat.py`

Adds `/v1/chat/completions` (with SSE streaming support) and `/v1/models` routes to the
existing Windows FastAPI app.  The shim:

- Converts OpenAI message format → internal GemmaService format (and back)
- Supports `stream: true` with SSE and `stream_options.include_usage`
- Supports `top_k` via `extra_body` (non-standard but vLLM-compatible)
- Is registered automatically in `app.py` via `register_openai_routes()`

### 2. WSL2 environment

- **Already installed:** Ubuntu 22.04, WSL2 version 2, RTX 3090 GPU pass-through working
- **venv:** `~/vllm-env` with vLLM 0.19.0, PyTorch 2.10.0+cu128, CUDA available
- **`setup_vllm.sh`:** Updated to create/reuse `~/vllm-env` automatically
- **`start.sh`:** Updated to activate `~/vllm-env` before launching vLLM

All vLLM scripts now live in `vllm-serving/` (sibling to `model-serving/`).

### 3. Tests

- `model-serving/tests/test_openai_compat.py` — 7 new tests covering message conversion
  and response building (no GPU required)
- All 12 UI tests pass unchanged (UI doesn't know which backend is running)
- 14 model-serving planning/benchmarking tests pass (no service dependency)

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────┐
│                    Streamlit UI                       │
│               (Windows, always the same)              │
│                                                       │
│   serving_client.py → POST /v1/chat/completions      │
│                     → GET  /v1/models                 │
│                     → GET  /health                    │
└───────────────────────┬─────────────────────────────┘
                        │  HTTP (localhost:8000)
            ┌───────────┴────────────┐
            │                        │
   ┌────────▼──────────┐   ┌────────▼──────────┐
   │   Mode 1: vLLM    │   │   Mode 2: Windows │
   │   (WSL2/Linux)    │   │   (FastAPI shim)   │
   │                    │   │                    │
   │  start_vllm.ps1   │   │  start_server.ps1  │
   │  → start.sh       │   │  → uvicorn app:app │
   │  → vllm serve     │   │  → GemmaService    │
   │                    │   │  → openai_compat   │
   │  Native OpenAI    │   │  OpenAI shim over  │
   │  endpoints        │   │  /generate          │
   └────────────────────┘   └────────────────────┘
```

---

## File Inventory

| File | Purpose | New/Modified |
|---|---|---|
| `model-serving/src/gemma_serving/openai_compat.py` | OpenAI API shim for Windows backend | New |
| `model-serving/src/gemma_serving/app.py` | Wires shim via `register_openai_routes()` | Modified (prior session) |
| `vllm-serving/setup_vllm.sh` | One-time WSL2 vLLM installer (creates ~/vllm-env) | Modified |
| `vllm-serving/start.sh` | vLLM launcher — now activates ~/vllm-env first | Modified |
| `vllm-serving/start_vllm.ps1` | Windows → WSL2 bridge | Unchanged |
| `model-serving/start_server.ps1` | Windows-native FastAPI launcher | Unchanged |
| `model-serving/tests/test_openai_compat.py` | 7 unit tests for the shim | New |

---

## How To Use Each Mode

### Mode 1: vLLM (recommended for benchmarks and Mistral)

```powershell
# One-time setup (from Windows):
wsl -d Ubuntu-22.04 -- bash -c "cd /mnt/c/Users/$env:USERNAME/source/repos/ai-workbench/vllm-serving && bash setup_vllm.sh"

# Start vLLM server:
cd vllm-serving
.\start_vllm.ps1                                    # default: Gemma 4 E2B
.\start_vllm.ps1 -Model "solidrust/Mistral-Small-3.1-24B-Instruct-2503-AWQ"  # Mistral

# Start UI (separate terminal):
cd ui
$env:PYTHONPATH="src"; streamlit run app.py
```

### Mode 2: Windows-native (quick iteration)

```powershell
# Start the existing FastAPI server:
cd model-serving
python start_server.py          # or .\start_server.ps1

# Start UI (separate terminal):
cd ui
$env:PYTHONPATH="src"; streamlit run app.py
```

Both modes serve on `localhost:8000` and the UI connects automatically.

---

## Trade-offs

| Dimension | vLLM (Mode 1) | Windows-native (Mode 2) |
|---|---|---|
| Performance | Best (PagedAttention, continuous batching) | Good (single-request, manual OOM guard) |
| Mistral support | Official, correct | "Not thoroughly tested" per Mistral AI |
| Setup complexity | WSL2 + venv in Linux | Just pip install in Windows venv |
| Streaming | Native SSE | Threaded SSE via shim |
| New model support | `MODEL_ID=... ./start.sh` (in vllm-serving/) | May need code changes |
| Quantization | AWQ, GPTQ built-in | BitsAndBytes NF4 only |
| Multi-GPU | Tensor parallelism built-in | Not supported |

**Recommendation:** Use vLLM (Mode 1) for anything beyond quick UI prototyping.
Use Windows-native (Mode 2) when you just want to test the UI without WSL2.

---

## WSL2 Environment Summary

| Component | Value |
|---|---|
| Distribution | Ubuntu 22.04 |
| WSL version | 2 |
| System Python | 3.10.12 |
| vLLM venv | `~/vllm-env` |
| vLLM version | 0.19.0 |
| PyTorch | 2.10.0+cu128 |
| CUDA | Available (RTX 3090, 24 GB) |
| GPU pass-through | Working (`nvidia-smi` sees the card) |
