# Purpose

This document tracks project phases, task hierarchy, acceptance criteria, and discovered follow-up work.
It exists so humans and AI agents can coordinate implementation without depending on chat history.
AI agents should read this before implementation and update it after any meaningful work.

# Maintenance Instructions

Update this file whenever a task starts, changes scope, completes, or reveals follow-up work.
Humans or AI may update it.
This file should stay aligned with [docs/design/design.md](./design/design.md) and reflect only the current known plan.

## Phase 1 - Bootstrap And Concept Validation

### 1.1 Create documentation-first project scaffold
- Status: [x]
- Started:
- Completed:
- Included in version: 0.1.0
- Acceptance criteria:
  - Repository includes the required documentation-first structure.
  - Repository includes a conventional unit test location when the chosen stack has one.
  - Each required Markdown file begins with purpose and maintenance guidance.
  - Restart instructions exist for future AI sessions.
- Validation: File structure created in repo.
- Notes: Missing business inputs are marked as To Be Discovered. Python projects in this repo should default to a top-level tests folder unless architecture requires otherwise.
- Dependencies: None

### 1.2 Choose a strong Gemma 4 sandbox usage
- Status: [x]
- Started:
- Completed:
- Included in version: 0.1.0
- Acceptance criteria:
  - Sandbox concept fits Gemma 4 native capabilities.
  - Unsupported generation modes are represented honestly.
  - Concept is documented in design and research notes.
- Validation: Concept documented as Multimodal Situation Room.
- Notes: Text-to-image, text-to-video, and text-to-audio are simulated planning modes, not native media synthesis.
- Dependencies: 1.1

### 1.3 Build starter sandbox UI
- Status: [x]
- Started:
- Completed:
- Included in version: 0.1.0
- Acceptance criteria:
  - User can select among requested abilities.
  - UI clearly distinguishes real, experimental, and simulated workflows.
  - UI accepts prompt input and file uploads where relevant.
- Validation: Streamlit app scaffold added.
- Notes:
- Dependencies: 1.1, 1.2

### 1.4 Encapsulate Gemma 4 access in reusable module
- Status: [x]
- Started:
- Completed:
- Included in version: 0.1.0
- Acceptance criteria:
  - Model access is isolated from the UI.
  - Standardized sampling defaults are applied.
  - Ability orchestration logic can be reused by another app.
- Validation: Reusable service modules now live under model-serving/src/gemma_serving (previously src/gemma_sandbox).
- Notes:
- Dependencies: 1.2

### 1.5 Validate local runtime setup
- Status: [ ]
- Started:
- Completed:
- Included in version:
- Acceptance criteria:
  - Dependencies install cleanly.
  - Streamlit app starts.
  - At least one Gemma call succeeds on target hardware.
- Validation:
- Notes: Streamlit watcher noise from Transformers lazy imports is mitigated by disabling file watching in .streamlit/config.toml. Stable Transformers 5.5.0 requires Gemma4ForConditionalGeneration or AutoModelForMultimodalLM instead of AutoModelForConditionalGeneration. Hardware and model download are still not fully exercised.
- Dependencies: 1.3, 1.4

### 1.6 Add runtime progress visibility
- Status: [x]
- Started:
- Completed:
- Included in version: 0.1.1
- Acceptance criteria:
  - Long-running model startup shows clear UI progress states.
  - Server logs reflect major runtime phases.
  - Cold-start behavior is explained in the UI.
- Validation: App emits progress for runtime check, processor load, model load, input prep, generation, token streaming, and decoding.
- Notes: Fine-grained download byte progress is still provided by Hugging Face tooling rather than a custom progress bridge. Text generation now streams partial output so long runs no longer look silent. Completed runs now surface response time, cold-start versus warm-start state, input/output/total token counts, and output tokens per second.
- Dependencies: 1.3, 1.4

### 1.8 Add selectable Gemma model profiles
- Status: [x]
- Started:
- Completed:
- Included in version: 0.1.6
- Acceptance criteria:
  - The UI offers multiple curated Gemma 4 model options.
  - Users can still supply a custom model ID.
  - Completed runs expose the active model ID in diagnostics.
- Validation: Sidebar now exposes official Gemma 4 instruction-tuned checkpoints plus a custom model option, and the run metadata panel includes the active model ID.
- Notes: Curated options currently use the official Hugging Face instruction-tuned IDs for E2B, E4B, 26B A4B, and 31B.
- Dependencies: 1.3, 1.4

### 1.9 Add richer run diagnostics
- Status: [x]
- Started:
- Completed:
- Included in version: 0.1.7
- Acceptance criteria:
  - Completed runs show separate runtime-load and generation timing.
  - Completed runs show prompt and response character counts.
  - Completed runs show approximate RAM or VRAM usage suitable for local model comparison.
- Validation: Service now attaches timing, character-count, and approximate memory metadata; the UI and standalone demo surface these diagnostics.
- Notes: CPU memory is approximated from process RSS, while CUDA memory uses Torch allocation and peak allocation counters when available.
- Dependencies: 1.4, 1.6, 1.8

### 1.10 Add response mode and custom system prompt controls
- Status: [x]
- Started:
- Completed:
- Included in version: 0.1.8
- Acceptance criteria:
  - The UI provides a checkbox to switch text generation between streaming and one-shot delivery.
  - One-shot mode disables token-by-token UI updates and uses a non-streaming generation path.
  - The user can edit the system prompt directly in the UI.
- Validation: Sidebar now includes a system prompt editor and a streaming toggle; the text service has distinct streaming and one-shot generation paths covered by tests.
- Notes: Simulator presets still shape the task prompt, while the new system prompt field controls the system role sent to Gemma.
- Dependencies: 1.3, 1.4, 1.8

### 1.11 Add conversation mode in the UI
- Status: [x]
- Started:
- Completed:
- Included in version: 0.1.9
- Acceptance criteria:
  - The UI allows follow-up questions without losing prior text context.
  - Conversation history can be cleared from the interface.
  - The runtime sends prior turns back to Gemma instead of only displaying them locally.
- Validation: Added Streamlit session-backed conversation history for text-capable abilities and passed prior turns into the sandbox service.
- Notes: Upload-based modes remain stateless for now because conversation-safe media reattachment is a separate concern.
- Dependencies: 1.3, 1.4, 1.10

### 1.12 Improve response placement in the UI
- Status: [x]
- Started:
- Completed:
- Included in version: 0.1.10
- Acceptance criteria:
  - Assistant replies appear adjacent to the triggering prompt or inside the active conversation thread.
  - The Result section focuses on diagnostics and metadata rather than duplicating the full response body.
- Validation: Conversation mode rerenders the updated chat thread inline, streams the in-progress assistant reply directly under the submitted question, and non-conversation runs show a Latest exchange block in the left column while the Result section remains diagnostic-focused.
- Notes: This also fixes the non-conversation Run button placement so it appears for all non-conversation abilities, not only video mode.
- Dependencies: 1.3, 1.10, 1.11

### 1.7 Add environment bootstrap
- Status: [x]
- Started:
- Completed:
- Included in version: 0.1.2
- Acceptance criteria:
  - Repository includes a local `.env` for runtime settings.
  - The app loads environment settings automatically on startup.
  - Python path overrides and model-related environment settings are centralized in environment config.
- Validation: Added `.env`, `.env.example`, and `env_bootstrap.py`; app and tests load env config before package imports.
- Notes: `.env` is gitignored while `.env.example` remains tracked as the shareable template.
- Dependencies: 1.1, 1.3, 1.4

## Phase 2 - Capability Hardening

### 2.1 Add prompt presets and sample missions
- Status: [ ]
- Started:
- Completed:
- Included in version:
- Acceptance criteria:
  - Each ability has example tasks.
  - Presets reflect real operator workflows.
- Validation:
- Notes:
- Dependencies: 1.3

### 2.4 Add portable demo script
- Status: [x]
- Started:
- Completed:
- Included in version: 0.1.3
- Acceptance criteria:
  - Repository includes a one-file Gemma 4 text demo script.
  - Script can be copied independently for colleague demos.
  - Script uses the documented Gemma 4 text flow and standardized sampling parameters.
- Validation: Added tests/gemma4_text_demo.py.
- Notes: The script is intentionally standalone and does not depend on the main Streamlit app. The UI and standalone demo now both report per-run response time.
- Dependencies: 1.4

### 2.5 Add production serving research pack
- Status: [x]
- Started:
- Completed:
- Included in version: 0.1.4
- Acceptance criteria:
  - Repository includes a dedicated research document for Gemma 4 model selection and serving tradeoffs.
  - Research captures planning assumptions for latency, concurrency, and cost of goods.
  - Repository archives the copied Gemma 4 source material under docs instead of the repo root.
- Validation: Added docs/research/gemma4-serving-evaluation.md and docs/research/sources/gemma4-model-card.md; removed obsolete root-level scratch files.
- Notes: Current latency and concurrency values are planning estimates only and must be replaced with measured benchmarks before infrastructure commitments.
- Dependencies: 1.4

### 2.6 Add serving research sibling package
- Status: [x]
- Started:
- Completed:
- Included in version: 0.1.5
- Acceptance criteria:
  - Repository includes a sibling package next to `src/gemma_sandbox` for serving and benchmark research utilities.
  - The new package contains small reusable planning helpers for concurrency and cost analysis.
  - Basic automated tests cover the initial planning helpers.
- Validation: Added `src/gemma_serving_research` and `tests/test_serving_research.py`.
- Notes: This package is intended for research and planning utilities, not end-user sandbox UI behavior.
- Dependencies: 2.5

### 2.7 Add research runners and low-cost FastAPI blueprint
- Status: [x]
- Started:
- Completed:
- Included in version: 0.1.6
- Acceptance criteria:
  - Repository includes a benchmark runner under `src/gemma_serving_research`.
  - Repository includes an E2B versus E4B concurrency simulation script.
  - Repository includes a concrete low-cost FastAPI serving blueprint that is separate from the Streamlit sandbox app.
- Validation: Added benchmark runner, simulation module, low-cost FastAPI blueprint code, focused tests, and supporting research documentation.
- Notes: The FastAPI blueprint currently uses a stub gateway and bounded in-memory queue by design; real Gemma inference should be added behind the same interface in a later task.
- Dependencies: 2.6

### 2.8 Add real Gemma gateway for the FastAPI blueprint
- Status: [x]
- Started:
- Completed:
- Included in version: 0.1.7
- Acceptance criteria:
  - Repository includes a real Gemma-backed gateway that matches the low-cost FastAPI blueprint contract.
  - The FastAPI blueprint can choose stub or real Gemma gateway through configuration.
  - Focused tests cover the gateway behavior without requiring model downloads.
- Validation: Added `src/gemma_serving_research/gemma_gateway.py`, wired env-based gateway selection, and added focused gateway tests.
- Notes: Stub remains the default to avoid accidental heavyweight startup; set `GEMMA_FASTAPI_GATEWAY=gemma` to use the real gateway.
- Dependencies: 2.7

### 2.9 Add real rewrite benchmark target and realistic scenarios
- Status: [x]
- Started:
- Completed:
- Included in version: 0.1.8
- Acceptance criteria:
  - Repository includes a real benchmark target that calls Gemma-backed listing rewrite.
  - Repository includes realistic eBay-style scenario inputs for E2B and E4B rewrite runs.
  - Focused tests validate benchmark-target metadata handling without requiring model downloads.
- Validation: Added `src/gemma_serving_research/benchmark_targets.py`, `docs/research/scenarios/ebay-listing-benchmarks.json`, and focused target tests.
- Notes: These scenarios are intended for actual local benchmark runs and will incur real model load and inference time when used with the real target.
- Dependencies: 2.8

### 2.2 Add structured outputs for simulated media workflows
- Status: [ ]
- Started:
- Completed:
- Included in version:
- Acceptance criteria:
  - Text-to-image returns reusable prompt packs.
  - Text-to-video returns storyboard and shot plan.
  - Text-to-audio returns narration and voice direction.
- Validation:
- Notes:
- Dependencies: 1.4

### 2.3 Add test coverage for prompt orchestration
- Status: [x]
- Started:
- Completed:
- Included in version: 0.1.1
- Acceptance criteria:
  - Core prompt-building and ability classification logic are covered.
  - Simulation modes are verified as non-native outputs.
- Validation: Starter prompt tests and a mocked simple text-generation service test added under tests/.
- Notes: Coverage now includes prompt construction, labeling, and a lightweight text-generation path test using fakes instead of real model weights.
- Dependencies: 1.4

## Discovered Work

### 3.1 Split repo into three-project architecture
- Status: [x]
- Started:
- Completed:
- Included in version: 0.2.0
- Acceptance criteria:
  - `model-serving/` is a standalone FastAPI project with its own requirements.txt, tests, and PYTHONPATH root.
  - `ui/` is a standalone Streamlit project that calls model-serving over HTTP via `ServingClient`.
  - `playground/` holds standalone demo and benchmark scripts.
  - Old flat `src/`, `tests/`, root `app.py`, root `requirements.txt` are removed.
  - Both test suites pass independently (24 serving, 4 UI).
  - Model-serving exposes `POST /generate` for generic text/multimodal inference.
  - UI has no torch/transformers dependency.
- Validation: Both test suites pass. Root structure is clean with only model-serving/, ui/, playground/, docs/, .github/.
- Notes: The serving package was renamed from `gemma_serving_research` to `gemma_serving`. The UI uses httpx `ServingClient` to call the FastAPI backend.
- Dependencies: 2.6, 2.7, 2.8, 2.9

### 3.2 Add root README with hardware recommendations
- Status: [x]
- Started:
- Completed:
- Included in version:
- Acceptance criteria:
  - Root README.md exists with project overview, repo layout, and quick start.
  - Measured CPU-only benchmark results are documented.
  - GPU tier recommendations for E2B, E4B, and larger models are documented.
  - Cost-effective strategies and minimum hardware summary are included.
- Validation: README.md created at repo root with corrected model sizes and verified GPU performance.
- Notes: Hardware recommendations updated with real measurements including RTX 3090 performance (7.4 tokens/sec for E2B). Corrected model parameter counts: E2B is 5.1B params, not 2B as naming suggests.
- Dependencies: 2.9

### 3.3 CUDA Performance Investigation and Optimization
- Status: [x]
- Started: 2026-04-04
- Completed: 2026-04-04
- Included in version: 0.2.1
- Acceptance criteria:
  - PyTorch CUDA installation verified and optimized for RTX 3090.
  - Actual Gemma 4 model sizes and performance characteristics measured and documented.
  - README updated with correct performance expectations based on real measurements.
  - CUDA optimization recommendations implemented in model serving code.
  - Requirements.txt updated with CUDA-enabled PyTorch versions.
- Validation: 
  - RTX 3090 performance verified: 7.4 tokens/sec for Gemma 4 E2B (5.1B parameters)
  - Model server includes CUDA availability warnings and optimization settings
  - README updated with correct model sizes and verified performance data
  - Installation scripts provide CUDA PyTorch setup instructions
- Notes: 
  - Discovered Gemma 4 E2B is actually 5.1B parameters, not 2B as naming suggests
  - Performance expectations in original README were incorrect for actual model sizes
  - PyTorch cuDNN benchmark optimization provides measurable performance improvement
  - Created comprehensive diagnostic tools for CUDA troubleshooting
- Dependencies: 3.1, 3.2

### 9.1 Add persistence for run history
- Status: [ ]
- Started:
- Completed:
- Included in version:
- Acceptance criteria:
  - The app can save past prompts and responses locally.
- Validation:
- Notes: Useful once real user flows are stable.
- Dependencies: 1.3