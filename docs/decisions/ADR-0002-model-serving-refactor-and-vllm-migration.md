# ADR-0002 — Model-Serving Refactor and vLLM Migration Plan

**Status:** IN PROGRESS — Phase B partially executed (UI side complete, WSL2 validation pending)  
**Date:** 2026-04-05 (proposed) / 2026-04-06 (Phase B UI work executed)  
**Author:** Copilot (review with human before executing)

---

## Context

`model-serving/` was built specifically for Gemma 4 using the Hugging Face `transformers` library.
Every layer reflects this: the Python package is named `gemma_serving`, the service class is `GemmaService`,
all config env-vars are `GEMMA_*`, and model-specific quirks (e.g. `fix_mistral_regex` for Mistral) are
handled with inline `if "mistral" in model_id` branches inside the service.

Two problems this creates:

1. **Adding any new model requires touching core service code.** There is no clean extension point.
2. **The hand-rolled inference stack (~600 lines in `gemma_service.py`) reimplements things vLLM
   already does better:** memory management, batching, streaming, multi-model routing.

Additionally, the Mistral Small 3.1 model card explicitly states:
> *"the transformers implementation was not thoroughly tested… we can only ensure 100% correct behavior when using vLLM."*

---

## Decision

Execute in two sequential phases:

- **Phase A** — Rename and restructure `model-serving/` to be model-agnostic (no vLLM yet). Low risk, backward compatible.
- **Phase B** — Replace the Transformers inference stack with vLLM as the serving backend. Higher impact, requires WSL2. Gated on playground validation.

---

## Phase A: Model-Agnostic Refactor (Transformers backend kept)

### A.1 Rename the Python package

```
model-serving/src/gemma_serving/  →  model-serving/src/model_serving/
gemma_service.py                  →  model_service.py
class GemmaService                →  class ModelService
GemmaLowCostGateway               →  ModelGateway
```

Keep a one-line backward-compat alias in `gateway.py`:
```python
GemmaLowCostGateway = ModelGateway  # backward compat
```

Update every `from gemma_serving.X import Y` inside the package to `from model_serving.X import Y`.

### A.2 Rename config env-vars (backward compatible)

Current env-var → new env-var (old name still works as fallback):

| Old | New |
|---|---|
| `GEMMA_MODEL_ID` | `MODEL_ID` |
| `GEMMA_QUANTIZE_4BIT` | `MODEL_QUANTIZE_4BIT` |
| `GEMMA_MODEL_CACHE_DIR` | `MODEL_CACHE_DIR` |
| `GEMMA_MAX_INPUT_TOKENS` | `MODEL_MAX_INPUT_TOKENS` |
| `GEMMA_FASTAPI_GATEWAY` | `MODEL_GATEWAY` |
| `GEMMA_FORCE_CPU` | `MODEL_FORCE_CPU` |
| `GEMMA_DEVICE_MAP` | `MODEL_DEVICE_MAP` |

Implementation in `config.py`:
```python
DEFAULT_MODEL_ID = os.getenv("MODEL_ID") or os.getenv("GEMMA_MODEL_ID", "google/gemma-4-E2B-it")
```

Existing `.env` files need no changes.

### A.3 Centralise model quirks into a new file

New file: `model_serving/model_quirks.py`

```python
def processor_kwargs(model_id: str) -> dict:
    """Per-model kwargs to pass to AutoProcessor.from_pretrained."""
    if "mistral" in model_id.lower():
        return {"fix_mistral_regex": True}   # see HF discussion #84
    return {}

def suggested_max_tokens(model_id: str) -> int:
    """Safe default context ceiling per model family."""
    if "mistral" in model_id.lower():
        return 32768
    return 8192  # Gemma 4 default
```

`_ensure_processor` in `model_service.py` calls `processor_kwargs(model_id)` instead of the
inline `if "mistral" in model_id` branch. Adding support for a new model quirk now only
requires editing `model_quirks.py`.

### A.4 Update app.py cosmetics

- FastAPI title: `"Gemma Model Serving"` → `"Model Serving"`
- Startup banner: `"GEMMA MODEL SERVING"` → `"MODEL SERVING"`
- `_get_gemma_service()` → `_get_model_service()`

### A.5 Update start_server.py

- Uvicorn app target: `gemma_serving.app:app` → `model_serving.app:app`
- `PYTHONPATH` comment updated

### A.6 Update and rename tests

- `tests/test_gemma_service.py` → `tests/test_model_service.py`
- Fix broken test `test_generate_text_matches_gemma_getting_started_flow`: it monkeypatches
  `AutoModelForCausalLM` but the live code calls `AutoModelForMultimodalLM` — patch the right symbol
- All `from gemma_serving.X` → `from model_serving.X`

### A.7 Update benchmark_targets.py

`_model_id_from_label` currently only resolves Gemma label strings.
Add: if `model_label` contains `/` treat it as a literal HF model ID (pass through).
Keep the existing Gemma label→ID table as a fallback.

### Phase A verification

```powershell
$env:PYTHONPATH="model-serving/src"
.\venv\Scripts\python.exe -m pytest model-serving/tests/ -v
# All tests pass

cd model-serving
python start_server.py
# Server starts, /health returns 200
```

Existing `.env` using `GEMMA_*` vars still works.
UI is unaffected — it only calls HTTP endpoints.

---

## Phase B: vLLM Backend (replaces Transformers inference)

**Prerequisite:** The two playground scripts must pass first:
```
playground/vllm_gemma4.py    # Gemma 4 E2B via vLLM server
playground/vllm_mistral.py   # Mistral Small 3.1 24B AWQ via vLLM server
```

### Why vLLM

| Capability | Transformers (current) | vLLM |
|---|---|---|
| Memory management | Manual OOM guard + truncation | PagedAttention — no OOM |
| Concurrent users | Single-threaded queue worker | Continuous batching built-in |
| Streaming | Manual `TextIteratorStreamer` thread | Native SSE |
| API contract | Custom `/generate` schema | OpenAI `/v1/chat/completions` |
| Adding a new model | Code changes needed | `vllm serve <model_id>` |
| Mistral correctness | transformers marked "untested" by Mistral | Officially recommended |
| Windows native | Yes | WSL2 required |

### B.1 What gets deleted

The entire `model_service.py` (~600 lines) is deleted.
vLLM replaces it with a single shell command.

### B.2 New model-serving structure

```
model-serving/
  models/
    gemma-4-e2b.yaml         # per-model vLLM launch config
    mistral-small-3.1.yaml
  start.ps1                  # Windows: opens WSL2 + runs start.sh
  start.sh                   # WSL2/Linux: launches vllm serve from models/*.yaml
  src/
    model_serving/
      app.py                 # thin FastAPI shim (optional — see B.3)
      config.py              # model registry + port config
      gateway.py             # listing/attribute logic unchanged
      domain.py              # unchanged
      benchmarking.py        # unchanged
      planning.py            # unchanged
      simulation.py          # unchanged
  tests/                     # all tests except test_model_service.py (deleted)
```

Example `models/gemma-4-e2b.yaml`:
```yaml
model_id: google/gemma-4-e2b-it
port: 8000
max_model_len: 8192
gpu_memory_utilization: 0.90
dtype: bfloat16
extra_args: []
```

Example `models/mistral-small-3.1.yaml`:
```yaml
model_id: mistralai/Mistral-Small-3.1-24B-Instruct-2503
port: 8000
max_model_len: 32768
gpu_memory_utilization: 0.92
quantization: awq
dtype: float16
extra_args: []
```

### B.3 API contract options (choose one)

**Option 1 — Drop the `/generate` wrapper, point directly at vLLM**
- UI's `serving_client.py` changes endpoint from `POST /generate` to `POST /v1/chat/completions`
- Response schema changes slightly (`choices[0].message.content` instead of `text`)
- Cleanest long-term

**Option 2 — Keep thin FastAPI shim**
- `app.py` stays, but its `/generate` handler becomes a proxy to vLLM's `/v1/chat/completions`
- UI needs zero changes
- Slightly more moving parts

Recommended: **Option 1** — the OpenAI schema is the industry standard and any future agent/client
will expect it.

### B.4 UI changes (Option 1 only)

File: `ui/src/gemma_sandbox/services/serving_client.py`

```python
# Before
POST /generate
body: {"messages": [...], "model_id": "...", "max_new_tokens": 256, ...}
response: {"text": "...", "input_token_count": ..., ...}

# After
POST /v1/chat/completions
body: {"model": "...", "messages": [...], "max_tokens": 256, ...}
response: {"choices": [{"message": {"content": "..."}}], "usage": {...}}
```

Only `serving_client.py` changes. `sandbox_service.py`, `domain.py`, `prompts.py`, `app.py` — all unchanged.

### B.5 start.ps1 (Windows launcher)

```powershell
# start.ps1
param([string]$Model = "gemma-4-e2b")

$config = Get-Content "models/$Model.yaml" | ConvertFrom-Yaml
wsl -e bash -c "vllm serve $($config.model_id) --host 0.0.0.0 --port $($config.port) ..."
```

### Phase B verification

```powershell
# Start vLLM server
.\model-serving\start.ps1 -Model gemma-4-e2b

# Run playground smoke test
python playground/vllm_gemma4.py

# Run UI tests (serving_client updated)
$env:PYTHONPATH="ui/src"
.\venv\Scripts\python.exe -m pytest ui/tests/ -v

# Run model-serving unit tests (gemma_service tests removed, rest intact)
$env:PYTHONPATH="model-serving/src"
.\venv\Scripts\python.exe -m pytest model-serving/tests/ -v
```

---

## What Does NOT Change

- `ui/app.py` — no change (except Option 1 requires `serving_client.py` update)
- `ui/src/gemma_sandbox/` — no change except `serving_client.py` (Option 1)
- `model-serving/src/model_serving/gateway.py` — listing/attribute logic unchanged
- `model-serving/src/model_serving/domain.py` — unchanged
- `model-serving/src/model_serving/benchmarking.py` — unchanged
- `model-serving/src/model_serving/planning.py` — unchanged
- `model-serving/src/model_serving/simulation.py` — unchanged
- HTTP API contract is preserved (or updated once cleanly in serving_client.py)
- All non-service tests continue to pass

---

## VRAM Requirements (RTX 3090 — 24 GB)

| Model | Format | VRAM | Works? |
|---|---|---|---|
| Gemma 4 E2B | bf16 | ~4 GB | ✅ |
| Gemma 4 E4B | bf16 | ~8 GB | ✅ |
| Gemma 4 27B | 4-bit AWQ | ~16 GB | ✅ |
| Mistral Small 3.1 24B | AWQ 4-bit | ~14 GB | ✅ |
| Mistral Small 3.1 24B | bf16 | ~55 GB | ❌ needs multi-GPU |

---

## Execution Order

1. ✅ (done) Write `playground/vllm_gemma4.py` and `playground/vllm_mistral.py`
2. ⬜ Set up WSL2 + install vLLM + run playground scripts → validate on RTX 3090
3. ⬜ Execute Phase A (package rename, model-agnostic refactor) — deferred; Phase B UI work done first
4. ✅ (done) Execute Phase B UI side — `serving_client.py` rewritten for OpenAI API (Option 1), `sandbox_service.py` updated, Mistral Small 3.1 enabled in model_profiles
5. ✅ (done) Created `model-serving/.env.vllm`, `start.sh`, `start_vllm.ps1`, `setup_vllm.sh`
6. ✅ (done) All 12 UI tests pass
7. ⬜ WSL2 end-to-end validation with vLLM server
8. ⬜ Archive or remove old `gemma_serving/` package after WSL2 validation
9. ⬜ Update `docs/design/design.md` and `docs/tasks.md` after each phase

---

## Resolved Open Questions

1. **Option 1 chosen.** The `/generate` wrapper is dropped. The UI talks directly to vLLM's OpenAI-compatible `/v1/chat/completions` endpoint. `serving_client.py` was fully rewritten.
2. **Default model: Gemma 4 E2B.** Smallest and fastest; set in `.env.vllm`.
3. **Quantization: user-configured via `.env.vllm`.** `QUANTIZATION=` (blank for small models), `QUANTIZATION=awq` for 24B+ models. No UI knob yet.
4. **Progress simplification: accepted.** vLLM does not expose per-stage callbacks. The UI uses SSE streaming for live token output; cold-start progress is replaced by a simple spinner.
