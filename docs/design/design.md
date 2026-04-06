# Purpose

This document is the top-level design source for the project.
It exists to capture the product intent, architecture, and current constraints in a form that humans and AI agents can reliably reuse.
AI agents must read this before implementing related tasks and update it after completing them.

# Maintenance Instructions

Update this file when project goals, architecture, supported abilities, or capability boundaries change.
Humans or AI may update it.
Keep this file as the design overview and create subdocuments later only if a domain becomes large enough to justify splitting.

## Purpose of this Document

This document is part of the modular project design documentation.
It captures the current understanding of this domain area.
Content may be incomplete early in the project.
Keep this document up-to-date as implementation evolves.
AI agents must read this before implementing related tasks and update it after completing them.

## Product Vision

The project is a multimodal AI sandbox — a game-like workspace for exploring model behaviors through a single operator console.
The chosen simulator concept is the Multimodal Situation Room.

The current implementation uses Gemma 4, which is good at:

- Text generation and reasoning
- Image understanding
- Audio transcription and understanding on E2B and E4B
- Long-context multimodal analysis
- Planning and structured output generation

Gemma 4 does not natively synthesize images, video, or audio. The sandbox must not pretend otherwise.

### Multi-Model Vision

The three-project architecture is intentionally model-agnostic at the API boundary. The UI talks to the serving backend through the OpenAI-compatible `/v1/chat/completions` contract and has no knowledge of which model or engine is loaded. This means:

- Swapping Gemma for another model (Llama, Phi, Mistral, Qwen, etc.) requires only changing the `MODEL_ID` in `.env.vllm` and restarting the vLLM server.
- The serving layer can host any model that vLLM supports — no code changes needed.
- The UI's model profiles, system prompt, and conversation state are model-independent.

Multi-model support is now live. `model_profiles.py` registers Gemma 4 (E2B, E4B, 26B A4B, 31B) and Mistral Small 3.1 (24B) with per-model capability flags (image, audio, video). The UI gates media input controls based on these capabilities.

## Sandbox Concept

The user enters a single conversation room — a persistent multimodal chat thread where any turn can
include an attached image, audio file, or video. There is no per-session ability or mode selector.
The model sees the full conversation history on every request.

The **system prompt** is a free-form text area that the user edits directly. There are no
pre-built persona presets — the user controls framing, role, and output format entirely
through the system prompt. This replaced the earlier persona dropdown.

## Users And Personas

- AI builders evaluating Gemma 4 locally
- Product designers exploring multimodal workflows
- Researchers testing prompt patterns across modalities
- Demo creators who need a safe, explainable sandbox

Some persona details are still To Be Discovered.

## Functional Requirements

- The UI always operates in conversation mode. Every turn contributes to a persistent thread passed to the model.
- The user may attach an image, audio file, or video to any turn. Media type is detected automatically from the file extension.

- The UI must let users switch among curated model profiles (Gemma 4 + Mistral) and optionally enter a custom model ID.
- The UI must let users edit the system prompt directly (free-form text area) and choose whether text responses stream or arrive in one shot.
- The app must route all model calls through `ServingClient`, which talks to the vLLM backend's OpenAI-compatible API.
- The app must apply the standardized sampling defaults:
  - temperature = 1.0
  - top_p = 0.95
  - top_k = 64
- The app must support a configurable max token limit.
- The UI must distinguish native and simulated output modes.
- The project must remain documentation-first.

## Non-Functional Requirements

- Be honest about capability boundaries.
- Keep the starter implementation easy to run locally.
- Keep the code modular enough to reuse the model service in another project.
- Prefer small, reviewable changes.

## Architecture

The repository is split into three independent projects under a single git root.

### Project Structure

```
ai-sandbox/
├── vllm-serving/           # vLLM launch scripts and config (WSL2/Linux only)
│   ├── .env.vllm           # vLLM server config (MODEL_ID, port, VRAM, quantization)
│   ├── start.sh            # Bash launcher for vLLM (WSL2/Linux)
│   ├── start_vllm.ps1      # Windows PowerShell wrapper (delegates to WSL2)
│   └── setup_vllm.sh       # One-time vLLM install script for WSL2
├── model-serving/          # Model-serving backend (Windows-native Transformers + planning)
│   ├── src/model_serving/  # Python package
│   │   ├── planning/       # Capacity planning, benchmarking, simulation subpackage
│   │   │   ├── planning.py
│   │   │   ├── simulation.py
│   │   │   ├── benchmarking.py
│   │   │   └── benchmark_targets.py
│   │   ├── app.py, config.py, domain.py, gateway.py
│   │   ├── model_service.py, openai_compat.py
│   │   └── __init__.py
│   ├── tests/              # Unit tests
│   └── requirements.txt
├── ui/                     # Streamlit frontend — no model weights, calls vLLM over HTTP
│   ├── src/ai_sandbox/  # Python package
│   │   ├── services/
│   │   │   ├── serving_client.py  # httpx client calling /v1/chat/completions (OpenAI API)
│   │   │   └── sandbox_service.py # Sandbox orchestration
│   │   ├── config.py       # AppConfig with serving_url
│   │   ├── model_profiles.py # Model registry with per-model capabilities
│   │   ├── domain.py, media.py, prompts.py
│   │   └── __init__.py
│   ├── tests/              # 12 tests
│   ├── app.py              # Streamlit entry point
│   ├── env_bootstrap.py
│   ├── .streamlit/config.toml
│   ├── requirements.txt    # streamlit, httpx, Pillow (no torch)
│   └── .env.example
├── playground/             # Standalone demo and benchmark scripts
│   ├── vllm_gemma4.py      # vLLM smoke test for Gemma 4
│   ├── vllm_mistral.py     # vLLM smoke test for Mistral Small 3.1
│   ├── gemma4_text_demo.py
│   ├── benchmark_runner.py
│   ├── concurrency_simulation.py
│   └── README.md
└── docs/                   # Shared project documentation
```

### Layers

1. vLLM layer (`vllm-serving/`)
   - vLLM runs as a standalone server inside WSL2 (or native Linux), launched via `start.sh`
   - Exposes the OpenAI-compatible API: `POST /v1/chat/completions`, `GET /v1/models`, `GET /health`
   - Supports any model vLLM can load — swap models by changing `MODEL_ID` in `.env.vllm`
   - Auto-detects Mistral models and applies `--tokenizer_mode mistral` flags
   - Memory management handled by vLLM’s PagedAttention — no manual OOM guard
   - Supports AWQ quantization for large models (24B+) on RTX 3090
   - Windows users launch via `start_vllm.ps1` which delegates to WSL2
   - Contains zero Python application code — only shell scripts and config

2. Model-serving layer (`model-serving/`)
   - Windows-native Transformers-based inference with OpenAI shim
   - `planning/` subpackage contains capacity planning, benchmarking, and simulation utilities
   - The old Transformers-based package was renamed from `gemma_serving` to `model_serving` (see ADR-0002)

2. Streamlit UI layer (`ui/`)
   - Collects output mode, prompt, and optional media attachment per turn
   - Calls the vLLM backend over HTTP via `ServingClient` (OpenAI-compatible API)
   - Has no model weights or torch dependency
   - Lets the user choose among curated model profiles (Gemma 4 + Mistral) or enter a custom model ID
   - Lets the user edit the system prompt used for the run
   - Lets the user switch text generation between streaming and one-shot delivery
   - Always operates in conversation mode — uses `st.chat_input` for all turns, shows a persistent multi-turn thread
   - Passes the full prior conversation (including media content parts) to the model on every request
   - Renders assistant replies inline with the prompt or chat thread
   - Shows runtime status, progress, and run metadata

3. Sandbox orchestration layer (`ui/src/ai_sandbox/services/sandbox_service.py`)
   - Maps abilities to native or simulated workflows
   - Builds prompts and message payloads
   - Applies capability labeling
   - Delegates inference to `ServingClient`

4. Media utilities layer (`ui/src/ai_sandbox/media.py`)
   - Stores uploads temporarily
   - Extracts representative video frames for analysis

5. Playground (`playground/`)
   - Standalone demo scripts for colleague handoff
   - Benchmark runner and concurrency simulation
   - No dependency on UI or serving packages

### Runtime Metadata

Each completed run should surface enough metadata to make local comparisons meaningful:

- Active model ID
- Cold-start versus warm-start state
- Response time in seconds
- Separate runtime-load, preparation, generation, and decode timings
- Input, output, and total token counts
- Prompt and response character counts
- Output tokens per second when token counts are available
- Approximate RAM or VRAM usage for the process or CUDA device
- Ability, preset, and generation settings used for the run

### Prompt Controls

- The system prompt is a free-form text area edited by the user directly. There are no persona presets.
- Text generation can run in streaming mode (SSE from vLLM) for live feedback or one-shot mode for a single final response.
- The conversation thread is always active and always sent to the model as `prior_turns`. Every turn’s
  media attachments are embedded in the history as content parts so multimodal follow-ups work naturally.
- Video files are sampled into representative frames before being sent; the frame paths are embedded in
  the history alongside the text so subsequent turns can reference the same frames.

### Starter Technology Choices

- Python for the application language
- Streamlit for the UI (in `ui/`)
- vLLM for model serving (runs in WSL2/Linux, exposes OpenAI-compatible API)
- httpx for UI-to-serving HTTP communication
- OpenCV for video frame extraction
- Pillow and SoundFile for local media handling
- Pytest with per-project test folders for unit tests
- Python dotenv for local environment configuration
- Standalone playground scripts for colleague handoff and quick local verification
- vLLM handles model loading, quantization (AWQ), PagedAttention memory management, and streaming — replacing the hand-rolled Transformers inference stack

## Capability Boundaries

### Native multimodal conversation (Chat output mode)

- Text generation and reasoning
- Image understanding — attach an image to any turn
- Audio transcription and understanding on E2B and E4B checkpoints — attach an audio file to any turn
- Video scene analysis through sampled frames — attach a video file to any turn; frames are auto-extracted

All media types can appear in the same conversation thread. Prior turns with media are re-sent to the
model so follow-up questions about previously uploaded files work correctly.

### Simulated planning modes (Plan Image / Plan Video / Plan Audio output modes)

These modes still use Gemma 4, but only for generating structured plans, prompts, scripts, or specs.
They do not produce pixels, audio waveforms, or rendered video.

## Edge Cases And Constraints

- Large models may not fit available local hardware.
- Audio support depends on using E2B or E4B checkpoints.
- Video support is bounded by frame extraction strategy and local file size.
- Thinking mode should remain optional because it increases latency and output size.
- Streamlit file watching can trigger noisy or failing lazy imports inside Transformers, so the starter app disables file watching in local configuration.
- Cold starts can be lengthy because processor download, model download, and weight loading may all happen on the first request. The UI and logs should report these stages clearly.
- Local runtime settings should live in `.env`, with `.env.example` as the tracked template. Python path additions should be applied through the environment bootstrap rather than scattered per entrypoint.
- Adding a second model should only require adding an entry to `model_profiles.py` and setting the `MODEL_ID` in `.env.vllm`. If a change forces UI modifications beyond model_profiles, the architecture needs redesigning first.
- The backend is dual-mode: vLLM in WSL2 (recommended) or Windows-native Transformers + OpenAI shim. Both expose the same `/v1/chat/completions` API. See [ADR-0003](../decisions/ADR-0003-dual-mode-serving.md).

## Testing Strategy

Each project has its own test folder and `PYTHONPATH` root:

- `model-serving/tests/` (24 tests, legacy — pending update for vLLM migration) — run with `PYTHONPATH=model-serving/src`
- `ui/tests/` (12 tests) — run with `PYTHONPATH=ui/src`

Tests focus on deterministic prompt-building, orchestration logic, model profile validation, and API contract validation using fakes and mocks rather than real model weights.

### Performance and Load Testing

Additional testing capabilities in `playground/`:

- **Sequential benchmarking** (`benchmark_runner.py`) — Single-threaded performance measurement with timing metrics
- **Concurrency simulation** (`concurrency_simulation.py`) — Mathematical capacity modeling for different user loads
- **Concurrent load testing** (`load_test.py`) — Multi-user stress testing with async HTTP clients

The load testing tool supports:
- 10-500+ concurrent users with configurable ramp-up
- Realistic production scenarios with SLA expectations
- Comprehensive metrics: throughput (RPS), latency percentiles (P50/P95/P99), error rates
- Bottleneck identification and capacity planning for production deployment

## Deployment And Operations

The current target is local prototype usage.  Two serving modes are supported
(see [ADR-0003](../decisions/ADR-0003-dual-mode-serving.md) for rationale):

### Mode 1: vLLM (recommended)

```powershell
# One-time WSL2 setup:
wsl -d Ubuntu-22.04 -- bash -c "cd /mnt/c/.../vllm-serving && bash setup_vllm.sh"

# Start vLLM:
cd vllm-serving
.\start_vllm.ps1                           # default model
.\start_vllm.ps1 -Model "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
```

vLLM config lives in `vllm-serving/.env.vllm`.

### Mode 2: Windows-native (quick iteration, no WSL2)

```powershell
cd model-serving
python start_server.py    # or .\start_server.ps1
```

Serves the same OpenAI-compatible endpoints via the built-in shim (`openai_compat.py`).

### UI (same for both modes)

```powershell
cd ui
$env:PYTHONPATH="src"; streamlit run app.py
```

The UI connects to `http://localhost:8000` by default (configurable via `MODEL_SERVING_URL` in `ui/.env`).

Serving research for potential multi-user deployment, including model tradeoffs, concurrency assumptions, and cost planning, is tracked in [docs/research/gemma4-serving-evaluation.md](../research/gemma4-serving-evaluation.md).