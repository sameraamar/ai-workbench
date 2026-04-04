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

The three-project architecture is intentionally model-agnostic at the API boundary. The UI talks to model-serving through a generic `POST /generate` contract (messages in, text out) and has no knowledge of which model is loaded. This means:

- Swapping Gemma for another HF model (Llama, Phi, Mistral, Qwen, etc.) requires changes only inside `model-serving/`, not in the UI.
- The serving layer could host multiple models behind the same endpoint, selected by a request field.
- The UI's ability modes, prompt presets, and conversation state are model-independent.

This is a design intent, not a current requirement. No multi-model code exists yet. When a second model is added, the serving config and loader will need generalization, but the API contract and UI should remain stable.

## Sandbox Concept

The user enters a single control room and chooses an ability:

- Text-to-text: native Gemma reasoning and generation
- Image-to-text: native Gemma image understanding
- Audio-to-text: native Gemma audio understanding on supported checkpoints
- Video-to-text: implemented as frame-based scene analysis with Gemma image understanding
- Text-to-image: simulated media-director mode that creates reusable prompt packs and creative specs
- Text-to-video: simulated storyboard mode that creates scene plans, shot lists, and pacing guidance
- Text-to-audio: simulated voice-production mode that creates narration scripts, tone guidance, and sound design notes

This makes the app useful immediately while leaving a clean seam for future integration with external media-generation models.

## Users And Personas

- AI builders evaluating Gemma 4 locally
- Product designers exploring multimodal workflows
- Researchers testing prompt patterns across modalities
- Demo creators who need a safe, explainable sandbox

Some persona details are still To Be Discovered.

## Functional Requirements

- The UI must let users choose the requested ability modes.
- The UI must let users switch among curated Gemma 4 model profiles and optionally enter a custom model ID.
- The UI must let users edit the system prompt and choose whether text responses stream or arrive in one shot.
- The UI always operates in conversation mode. Every ability shows a persistent thread with turn count and a Clear button.
- The app must route all Gemma calls through a reusable service module.
- The app must apply the standardized sampling defaults:
  - temperature = 1.0
  - top_p = 0.95
  - top_k = 64
- The app must support a configurable max token limit.
- The UI must distinguish native, experimental, and simulated abilities.
- The project must remain documentation-first.

## Non-Functional Requirements

- Be honest about capability boundaries.
- Keep the starter implementation easy to run locally.
- Keep the code modular enough to reuse the Gemma service in another project.
- Prefer small, reviewable changes.

## Architecture

The repository is split into three independent projects under a single git root.

### Project Structure

```
ai-sandbox/
├── model-serving/          # FastAPI backend — owns all model weights and inference
│   ├── src/gemma_serving/  # Python package
│   │   ├── app.py          # FastAPI application (POST /generate, job endpoints, /health)
│   │   ├── config.py       # ServingConfig dataclass
│   │   ├── gemma_service.py# Core Gemma 4 inference (text + multimodal)
│   │   ├── gateway.py      # GemmaLowCostGateway for marketplace operations
│   │   ├── domain.py, planning.py, simulation.py, benchmarking.py, benchmark_targets.py
│   │   └── __init__.py
│   ├── tests/              # 24 tests
│   ├── docs/scenarios/     # Benchmark scenario JSON files
│   ├── requirements.txt    # torch, transformers, fastapi, uvicorn (no streamlit)
│   └── .env.example
├── ui/                     # Streamlit frontend — no model weights, calls model-serving over HTTP
│   ├── src/gemma_sandbox/  # Python package
│   │   ├── services/
│   │   │   ├── serving_client.py  # httpx client calling POST /generate
│   │   │   └── sandbox_service.py # Sandbox orchestration
│   │   ├── config.py       # AppConfig with serving_url
│   │   ├── domain.py, media.py, prompts.py
│   │   └── __init__.py
│   ├── tests/              # 4 tests
│   ├── app.py              # Streamlit entry point
│   ├── env_bootstrap.py
│   ├── .streamlit/config.toml
│   ├── requirements.txt    # streamlit, httpx, Pillow (no torch)
│   └── .env.example
├── playground/             # Standalone demo and benchmark scripts
│   ├── gemma4_text_demo.py
│   ├── benchmark_runner.py
│   ├── concurrency_simulation.py
│   └── README.md
└── docs/                   # Shared project documentation
```

### Layers

1. Model-serving FastAPI layer (`model-serving/`)
   - Loads the model lazily on first `/generate` request (currently Gemma 4, but the `POST /generate` contract is model-agnostic)
   - Exposes `POST /generate` for generic text and multimodal inference
   - Exposes job-based endpoints for marketplace operations (rewrite, extract-attributes)
   - Applies standardized generation defaults
   - Supports 4-bit NF4 quantization via `BitsAndBytesConfig`
   - Supports `low_cpu_mem_usage` for faster weight loading
   - Reports timing, token counts, and memory metadata in response payloads

2. Streamlit UI layer (`ui/`)
   - Collects ability choice, prompt, and uploads
   - Calls model-serving over HTTP via `ServingClient`
   - Has no model weights or torch dependency
   - Lets the user choose among curated Gemma 4 checkpoints or enter a custom model ID
   - Lets the user edit the system prompt used for the run
   - Lets the user switch text generation between streaming and one-shot delivery
   - Always operates in conversation mode — uses `st.chat_input` for all abilities, shows a persistent multi-turn thread
   - Preserves prior turns in Streamlit session state; passes them to the model only for `MULTI_TURN_CAPABLE_ABILITIES` (text and simulated text modes)
   - Renders assistant replies inline with the prompt or chat thread
   - Shows runtime status, progress, and run metadata

3. Sandbox orchestration layer (`ui/src/gemma_sandbox/services/sandbox_service.py`)
   - Maps abilities to native or simulated workflows
   - Builds prompts and message payloads
   - Applies capability labeling
   - Delegates inference to `ServingClient`

4. Media utilities layer (`ui/src/gemma_sandbox/media.py`)
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

- Simulator presets are prompt-framing helpers that are prepended to the task prompt.
- The system prompt is a separate user-editable control that populates the system role sent to Gemma.
- Text generation can run in streaming mode for live feedback or one-shot mode for a single final response.
- Conversation mode is always active. Every ability shows a persistent thread with turn count and a Clear button. Text and simulated text abilities (`MULTI_TURN_CAPABLE_ABILITIES`) include prior turns in the model request. Media-upload abilities (image, audio, video) append turns to the visible thread but send each request in isolation because uploaded files cannot be re-attached across turns.

### Starter Technology Choices

- Python for the application language
- Streamlit for the UI (in `ui/`)
- FastAPI + Uvicorn for model serving (in `model-serving/`)
- httpx for UI-to-serving HTTP communication
- Hugging Face Transformers for Gemma integration
- PyTorch for model execution
- BitsAndBytesConfig for optional 4-bit NF4 quantization
- Torchvision for multimodal image-processing dependencies used by Transformers
- OpenCV for video frame extraction
- Pillow and SoundFile for local media handling
- Pytest with per-project test folders for unit tests
- Python dotenv for local environment configuration
- Standalone playground scripts for colleague handoff and quick local verification

## Capability Boundaries

### Native Gemma-backed modes

- Text-to-text
- Image-to-text
- Audio-to-text on E2B and E4B
- Video-to-text through sampled frames

### Simulated planning modes

- Text-to-image
- Text-to-video
- Text-to-audio

Simulated planning modes still use Gemma 4, but only for generating structured plans, prompts, scripts, or specs. They do not produce pixels, audio waveforms, or rendered video.

## Edge Cases And Constraints

- Large models may not fit available local hardware.
- Audio support depends on using E2B or E4B checkpoints.
- Video support is bounded by frame extraction strategy and local file size.
- Thinking mode should remain optional because it increases latency and output size.
- Streamlit file watching can trigger noisy or failing lazy imports inside Transformers, so the starter app disables file watching in local configuration.
- Cold starts can be lengthy because processor download, model download, and weight loading may all happen on the first request. The UI and logs should report these stages clearly.
- Local runtime settings should live in `.env`, with `.env.example` as the tracked template. Python path additions should be applied through the environment bootstrap rather than scattered per entrypoint.
- Adding a second model should only require changes inside `model-serving/`. If a change forces UI modifications, the API contract needs redesigning first.

## Testing Strategy

Each project has its own test folder and `PYTHONPATH` root:

- `model-serving/tests/` (24 tests) — run with `PYTHONPATH=model-serving/src`
- `ui/tests/` (4 tests) — run with `PYTHONPATH=ui/src`

Tests focus on deterministic prompt-building, orchestration logic, and API contract validation using fakes and mocks rather than real model weights.

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

The current target is local prototype usage with two processes:

1. Start model-serving: `cd model-serving && PYTHONPATH=src uvicorn gemma_serving.app:app --host 0.0.0.0 --port 8000`
2. Start UI: `cd ui && PYTHONPATH=src streamlit run app.py`

The UI connects to model-serving at `http://localhost:8000` by default (configurable via `SERVING_URL` in `ui/.env`).

Serving research for potential multi-user deployment, including model tradeoffs, concurrency assumptions, and cost planning, is tracked in [docs/research/gemma4-serving-evaluation.md](../research/gemma4-serving-evaluation.md).