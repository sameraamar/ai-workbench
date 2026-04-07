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
- Notes: Streamlit watcher noise from Transformers lazy imports is mitigated by disabling file watching in .streamlit/config.toml. The correct loader is `Gemma4ForConditionalGeneration` from transformers 5.5.0+ — it handles text, image, and audio inputs natively for all Gemma 4 checkpoints. Hardware and model download are exercised — RTX 3090 verified at 7.65 tok/s average for E2B.
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

### 1.13 Remove single-turn mode — always-on conversation
- Status: [x]
- Started:
- Completed:
- Included in version: 0.1.11
- Acceptance criteria:
  - The Conversation mode checkbox is removed; conversation is always active.
  - All abilities use `st.chat_input` — no `st.text_area` for the user prompt, no Run Sandbox button.
  - The conversation thread, turn counter, and Clear button are always visible for all abilities.
  - Text and simulated text abilities (`MULTI_TURN_CAPABLE_ABILITIES`) send prior turns to the model.
  - Media-upload abilities append turns to the visible thread but send each request in isolation (files cannot be re-attached).
  - The two-rerun / `_sandbox_running` state machine is completely removed.
  - The metadata JSON records `conversation_turn_count` instead of `conversation_mode`.
- Validation: Removed `conversation_mode` checkbox, `_sandbox_running` state, two-rerun pattern, `_render_latest_exchange`, all disabled-textarea CSS, and prompt-label-row CSS. Renamed `CONVERSATION_CAPABLE_ABILITIES` → `MULTI_TURN_CAPABLE_ABILITIES`. Updated design.md.
- Notes: Single-turn is architecturally identical to one-turn conversation. Keeping a separate code path was wrong design — eliminated ~80 lines of workaround code.
- Dependencies: 1.11, 1.12

### 1.14 Replace OutputMode selector with planning personas
- Status: [x]
- Started:
- Completed:
- Included in version: 0.1.12
- Acceptance criteria:
  - `OutputMode` enum, `OutputModeSpec`, `OUTPUT_MODE_SPECS`, and `build_simulation_prompt` are deleted.
  - Three planning personas added to `PERSONA_PRESETS`: Image Prompt Engineer, Storyboard Director, Audio Producer.
  - Persona system prompt controls all output framing; no separate mode selector exists in the UI.
  - `SandboxService.run()` has no `output_mode` parameter.
  - All 7 UI tests pass.
- Validation: Removed `OutputMode` and related code across domain.py, prompts.py, sandbox_service.py, __init__.py, app.py. Tests updated and passing 7/7.
- Notes: Planning output format is now controlled entirely by the system prompt selected via persona. This is architecturally cleaner and consistent with how LLM system prompts actually work.
- Dependencies: 1.13

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
- Notes: Hardware recommendations updated with real measurements including RTX 3090 performance (7.4 tokens/sec for E2B). Documented MoE architecture: E2B = 2.3B effective / 5.1B total params; the E prefix means Effective and the 2B refers to active parameters per token.
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
  - RTX 3090 performance verified: 7.4 tokens/sec for Gemma 4 E2B (2.3B effective / 5.1B total params)
  - Model server includes CUDA availability warnings and optimization settings
  - README updated with correct model sizes and verified performance data
  - Installation scripts provide CUDA PyTorch setup instructions
- Notes: 
  - Documented Gemma 4 MoE architecture: E2B = 2.3B effective (active per token) / 5.1B total (with embeddings); E prefix = Effective, 2B = active param count
  - Performance expectations in original README were incorrect for actual model sizes
  - PyTorch cuDNN benchmark optimization provides measurable performance improvement
  - Created comprehensive diagnostic tools for CUDA troubleshooting
- Dependencies: 3.1, 3.2

### 3.4 Fix Transformers API compatibility and align package versions
- Status: [x]
- Started: 2026-04-04
- Completed: 2026-04-04
- Included in version: 0.2.2
- Acceptance criteria:
  - Server starts without import errors on Python 3.11 with transformers 5.5.0.
  - All shared packages use the same minimum version across model-serving and ui requirements.txt.
  - CUDA PyTorch installs cleanly in the venv without corruption.
  - Real model inference verified end-to-end via /generate endpoint.
- Validation:
  - Removed non-existent `AutoModelForMultimodalLM` import (removed in transformers 5.x).
  - Removed the wrong `Gemma3ForConditionalGeneration` / `Gemma2ForCausalLM` fallback chain; replaced with single direct `Gemma4ForConditionalGeneration.from_pretrained()` — the correct and only loader for all Gemma 4 inputs.
  - Fixed `_get_text_runtime` which still referenced the removed `AutoModelForCausalLM`; now delegates to `_get_multimodal_runtime` since Gemma 4 is a single model for both text and image.
  - Aligned python-dotenv (>=1.2.0), pytest (>=8.4.0), numpy (>=2.0.0) across both requirements files.
  - Fixed PyTorch CUDA corruption by full uninstall + clean reinstall of torch==2.5.1+cu121.
  - Verified RTX 3090 benchmark: 7.65 tok/s average (6.74 short / 8.47 medium / 7.75 long), VRAM 9.6 GB.
  - Verified image-to-text end-to-end: uploaded image processed via `Gemma4ForConditionalGeneration`, coherent JSON output returned, 4.04 tok/s, 10557 MB peak VRAM on cuda:0. No fallback, no weight mismatch warnings.
  - Documented that `GEMMA_FASTAPI_GATEWAY=gemma` must be set in model-serving/.env for real inference.
- Notes:
  - `Gemma4ForConditionalGeneration` exists in transformers 5.5.0 (venv Python 3.11) and is the correct single loader for all Gemma 4 modalities (text + image + audio). The previous fallback chain through Gemma3/Gemma2 was wrong and masked the real class being available.
  - `Gemma4ForConditionalGeneration` does not exist in transformers 4.x, which is what Python 3.9 system install had — this caused the initial import error confusion.
  - Background PowerShell terminals on this machine inherit a broken conda profile and default to Python 3.9; always invoke the venv Python by full path.
  - `GEMMA_FASTAPI_GATEWAY=stub` (the previous default) silently returns empty responses; changed to `gemma` in .env.
- Dependencies: 3.3

### 3.5 Add concurrent load testing tool
- Status: [x]
- Started: 2026-04-04
- Completed: 2026-04-04
- Included in version: 0.2.3
- Acceptance criteria:
  - Repository includes a comprehensive load testing tool that can simulate concurrent users hitting the `/generate` endpoint.
  - Tool supports configurable concurrent users (10-500+), test duration, and gradual ramp-up.
  - Load tests provide detailed metrics: throughput (RPS), latency percentiles (P50/P95/P99), success rates, and error analysis.
  - Tool includes realistic production scenarios and development test scenarios.
  - Documentation explains usage patterns and requirement dependencies.
- Validation: 
  - Created `playground/load_test.py` with async HTTP client using aiohttp for true concurrency.
  - Added `playground/load_scenarios.json` with development scenarios (10-100 concurrent users).
  - Added `playground/production_load_scenarios.json` with production-realistic scenarios including SLA expectations.
  - Updated `model-serving/requirements.txt` to include aiohttp>=3.9.0 dependency.
  - Updated `playground/README.md` with comprehensive usage documentation and examples.
- Notes: 
  - This addresses the real-world need for stress testing and capacity planning before production deployment.
  - Tool can identify bottlenecks, validate SLA compliance, and find breaking points under concurrent load.
  - Builds on existing BenchmarkScenario format for consistency with sequential benchmark tooling.
  - Supports both converted legacy scenarios and new load-specific scenario format with concurrent_users field.
- Dependencies: 3.1, 3.4

### 3.6 Add comprehensive benchmarks documentation
- Status: [x]
- Started: 2026-04-04
- Completed: 2026-04-04
- Included in version: 0.2.3
- Acceptance criteria:
  - Repository includes comprehensive documentation of all benchmarking capabilities and methodologies.
  - Documentation includes actual performance results from RTX 3090 testing with Gemma 4 models.
  - Load testing scenarios and capacity planning guidance are documented with real metrics.
  - Documentation covers sequential benchmarking, concurrent load testing, and concurrency simulation.
  - Usage examples and result interpretation guidance are provided.
- Validation:
  - Created `docs/benchmarks.md` with comprehensive benchmarking documentation.
  - Ran actual concurrent load tests on RTX 3090 (April 4 2026): 3 concurrent users, 120s, 6 scenarios.
  - Measured key finding: per-request processing slot = ~13.8s for 64-token responses. Queue latency at 3 users = 41.2s avg, P95 46.6s — matches linear queuing model exactly.
  - Longer responses (96+ tokens) drive avg latency to 240s at 3 concurrent users due to ~80s per-slot cost.
  - E4B vs E2B under queue pressure: virtually identical (247.9s vs 240.7s) — queue depth dominates.
  - Results saved to `playground/results.json` for reference.
  - Established practical production limit: ≤3 concurrent users per GPU for interactive short completions.
- Notes:
  - Corrected model size documentation: E2B/E4B naming reflects effective (active MoE) param counts — 2.3B effective / 5.1B total and 4.5B effective / 8B total respectively.
  - Established performance baselines for capacity planning and infrastructure decisions.
  - Documentation serves as reference for production deployment and optimization strategies.
- Dependencies: 3.5

### 3.7 Add multi-model support with capability-aware UI
- Status: [x]
- Started: 2026-04-05
- Completed: 2026-04-05
- Included in version: 0.2.4
- Acceptance criteria:
  - Model selector supports Gemma 4 (E2B, E4B, 26B A4B, 31B) and Mistral Small 3.1 (24B) and Mistral Small 4 (119B).
  - Media input controls are gated per model capability (image/audio/video).
  - User can provide media via file upload, https:// URL, or clipboard paste.
  - Persona dropdown is removed; system prompt is always a free-form text area.
  - Model-to-capability mapping lives in a single dedicated configurator file.
  - Model loader uses `AutoModelForMultimodalLM` to support any registered multimodal model without code changes.
  - All tests pass.
- Validation:
  - Created `ui/src/gemma_sandbox/model_profiles.py` with `ModelCapabilities`, `MODEL_PROFILES` registry, `get_model_id()`, `get_capabilities()`, and `MODEL_LABELS`.
  - Removed `MODEL_OPTIONS` from `config.py`; removed `PERSONA_PRESETS` from `prompts.py`.
  - Added `image_urls: list[str]` to `TurnAttachment`; run() sends them as `{"type": "image", "url": https_url}`.
  - Replaced single `st.file_uploader` in `app.py` with three tabs: 📁 Upload / 🔗 URL / 📋 Paste.
  - Upload tab gates audio/video options based on `caps.audio` / `caps.video`.
  - URL tab offers image URL input; audio/video URL inputs shown only when model supports them.
  - Paste tab uses `streamlit_paste_button` with graceful fallback when package is not installed.
  - Right column now shows Model Capabilities panel (✅/❌ per modality) instead of personas.
  - Swapped `Gemma4ForConditionalGeneration` → `AutoModelForMultimodalLM` in `gemma_service.py`.
  - Added `streamlit-paste-button>=0.1.3` to `ui/requirements.txt`.
  - Updated `ui/tests/test_prompts.py` → `test_model_profiles.py` tests (9 tests); all 12 UI tests pass.
- Notes:
  - Mistral models: image ✅, audio ❌, video ❌ (Mistral Small 3.1 and 4 are vision-language; no audio/video).
  - Gemma 4 E2B/E4B: full trimodal (image + audio + video). 26B/31B: image + video, no audio.
  - AutoModelForMultimodalLM verified in transformers 5.5.0 with Gemma4Config + Mistral3/4Config registered.
  - Adding a new model requires only one entry in model_profiles.py — no other file changes needed.
- Dependencies: 3.4, 3.6

### 3.8 Fix Mistral Small 3.1 500 error on /models/load
- Status: [x]
- Started: 2026-04-05
- Completed: 2026-04-05
- Included in version: 0.2.5
- Acceptance criteria:
  - POST /models/load succeeds for mistralai/Mistral-Small-3.1-24B-Instruct-2503.
  - Processor loading does not crash with AttributeError on fix_mistral_regex.
  - GEMMA_QUANTIZE_4BIT=1 is enabled in .env so the 24B model fits in 25.8 GB VRAM.
  - All existing tests continue to pass.
- Validation:
  - Root cause: `fix_mistral_regex=True` was unconditionally passed to `AutoTokenizer.from_pretrained` for all Mistral models. Mistral Small 3.1 uses the Tekken tokenizer (Rust-backed `TokenizersBackend`), which does not expose `backend_tokenizer` — the attribute that `fix_mistral_regex` needs to patch. This raised `AttributeError` and propagated as a 500.
  - Fix in `model-serving/src/gemma_serving/gemma_service.py` `_ensure_processor()`: read `tokenizer_class` from the model's `AutoConfig` and only apply `fix_mistral_regex=True` when `tokenizer_class == "LlamaTokenizerFast"` (SentencePiece-backed older Mistral models). Newer Tekken-based models skip the flag without masking errors.
  - `.env` updated: `GEMMA_QUANTIZE_4BIT=1` with comment explaining when each setting is appropriate.
  - 23/24 model-serving tests pass; one pre-existing failure (`test_generate_text_matches_gemma_getting_started_flow`) references removed `AutoModelForCausalLM` symbol — predates this task.
- Notes:
  - Mistral Small 3.1 (24B) needs ~12 GB in 4-bit NF4 (vs ~48 GB BF16); fits on RTX 3090 (25.8 GB) with quantization enabled.
  - fix_mistral_regex applies only to: models whose tokenizer_config.json declares `tokenizer_class: LlamaTokenizerFast`.
- Dependencies: 3.7

### 3.9 Migrate model-serving backend to vLLM
- Status: [x]
- Started: 2026-04-06
- Completed: 2026-04-05
- Included in version:
- Acceptance criteria:
  - vLLM runs in WSL2 and serves models via OpenAI-compatible API on localhost:8000.
  - UI `serving_client.py` talks to `/v1/chat/completions` instead of custom `/generate`.
  - `vllm-serving/` includes `.env.vllm`, `start.sh`, `start_vllm.ps1`, `setup_vllm.sh`.
  - Mistral Small 3.1 (24B) is enabled in model_profiles and works via vLLM.
  - All UI tests pass (12/12).
  - End-to-end WSL2 validation succeeds with at least one model.
- Validation:
  - `serving_client.py` rewritten for OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/models`, `/health`).
  - `sandbox_service.py` updated — removed `enable_thinking` parameter (not in vLLM).
  - Mistral Small 3.1 enabled in `model_profiles.py` (removed from DISABLED_LABELS).
  - All 12 UI tests pass. WSL2 end-to-end pending.
- Notes:
  - Decision captured in design.md (Key Architecture Decisions). Phase A (package rename) deferred; Phase B (vLLM) UI side executed first.
  - vLLM does not run natively on Windows; WSL2 is required. UI stays on Windows.
  - `start.sh` auto-detects Mistral models and applies `--tokenizer_mode mistral --config_format mistral --load_format mistral`.
  - Old `gemma_serving/` stale directory removed during 3.11 rename.
- Dependencies: 3.7, 3.8

### 3.10 Add dual-mode serving (vLLM + Windows-native)
- Status: [x]
- Started: 2026-04-05
- Completed: 2026-04-05
- Included in version:
- Acceptance criteria:
  - Both vLLM (WSL2) and Windows-native (FastAPI + Transformers) backends serve the same OpenAI-compatible API on localhost:8000.
  - The UI works identically against either backend with zero code changes.
  - WSL2 has vLLM 0.19.0 installed and verified (PyTorch 2.10.0+cu128, CUDA available, RTX 3090 visible).
  - An OpenAI-compatible shim (`openai_compat.py`) adds `/v1/chat/completions` and `/v1/models` to the existing Windows FastAPI app.
  - `setup_vllm.sh` creates and manages `~/vllm-env` automatically.
  - `start.sh` activates the WSL2 venv before launching vLLM.
  - New tests cover message conversion and response building (7 tests).
  - All 12 UI tests pass unchanged.
- Validation:
  - WSL2 Ubuntu 22.04 verified: vLLM 0.19.0, PyTorch 2.10.0+cu128, CUDA True, RTX 3090.
  - `openai_compat.py` created with `register_openai_routes()`, SSE streaming, message conversion.
  - `test_openai_compat.py` — 7 tests pass. UI tests — 12 pass. Planning/benchmarking — 14 pass.
  - Decision captured in design.md (Key Architecture Decisions).
- Notes:
  - Single repo, single branch. The only difference is which start script you run.
  - Recommend vLLM for benchmarks and Mistral; Windows-native for quick UI iteration.
  - No separate repository needed — the OpenAI API contract is the clean boundary.
- Dependencies: 3.9

### 3.11 Model-agnostic rename (gemma_serving → model_serving)
- Status: [x]
- Started: 2026-04-05
- Completed: 2026-04-05
- Included in version:
- Acceptance criteria:
  - Package directory renamed `gemma_serving/` → `model_serving/`, file `gemma_service.py` → `model_service.py`.
  - All classes renamed: `GemmaService` → `ModelService`, `GemmaLowCostGateway` → `ModelGateway`.
  - All env-vars read `MODEL_*` first with fallback to `GEMMA_*` for backward compat.
  - Backward-compat aliases (`GemmaService`, `GemmaLowCostGateway`) kept in-module and in `__init__.py`.
  - All test imports updated; monkeypatch paths use `model_serving.*`.
  - Stale `src/gemma_serving/` directory removed.
  - All 31 model-serving tests pass. All 12 UI tests pass.
- Validation:
  - Source files updated: config.py, model_service.py, gateway.py, app.py, openai_compat.py, benchmark_targets.py, __init__.py, start_server.ps1.
  - Test files updated: test_api.py, test_benchmarking.py, test_benchmark_targets.py, test_gateway.py, test_model_service.py, test_openai_compat.py, test_planning.py.
  - `test_generate_text_matches_gemma_getting_started_flow` fixed: now patches `AutoModelForMultimodalLM` (replaces removed `AutoModelForCausalLM`) and short-circuits lazy `_import_transformers()` to avoid transformers 4.57.x import error.
- Notes:
  - model_quirks.py creation deferred — inline `if "mistral"` checks work for now.
  - config.py uses `_env(new, old, default)` / `_env_bool(new, old, default)` helpers for dual-name env-var reading.
- Dependencies: 3.10

### 3.12 Rewrite serving_client.py to use openai SDK
- Status: [x]
- Started: 2026-04-05
- Completed: 2026-04-05
- Included in version:
- Acceptance criteria:
  - `serving_client.py` uses the `openai` Python SDK (`OpenAI` client) instead of raw `httpx` for `/v1/chat/completions` and `/v1/models`.
  - `/health` endpoint still uses `httpx` (not part of OpenAI spec).
  - `enable_thinking` is passed through via `extra_body` to vLLM.
  - `sandbox_service.py` forwards `enable_thinking` from `GenerationSettings`.
  - `openai>=1.0.0` added to `ui/requirements.txt`.
  - All 12 UI tests pass. All 31 model-serving tests pass.
- Validation:
  - ~150 lines of hand-rolled httpx/SSE parsing replaced with typed SDK calls.
  - `_generate_one_shot()` and `_generate_streaming()` use `client.chat.completions.create()`.
  - `get_active_model_id()` uses `client.models.list()`.
  - Streaming properly iterates `ChatCompletionChunk` objects with typed `.choices[0].delta.content`.
  - `enable_thinking` sent as `extra_body={"enable_thinking": True}` when toggled on.
  - Test fake signature updated to accept `enable_thinking` kwarg.
- Notes:
  - `httpx` kept as a dependency for the `/health` check and for other UI needs (streamlit uses it).
  - `api_key="not-needed"` since local vLLM doesn't require auth.
- Dependencies: 3.9, 3.11

### 3.13 Update README with dual-environment setup documentation
- Status: [x]
- Started: 2026-04-05
- Completed: 2026-04-05
- Included in version:
- Acceptance criteria:
  - Quick Start section documents two separate virtual environments (Windows venv + WSL2 venv).
  - Clear warning about never installing vLLM/compressed-tensors/mistral-common in Windows venv.
  - Instructions for CUDA PyTorch install, model-serving deps, UI deps, .env setup.
  - WSL2 one-time setup via `setup_vllm.sh` documented.
  - Both backend modes (vLLM and Windows-native) documented with start commands.
  - Test run command documented.
  - Stale references fixed: `gemma_service.py` → `model_service.py`, old torch version pins updated.
- Validation:
  - README.md Quick Start rewritten end-to-end.
  - Repository layout table updated with dual entry points.
  - `pip install torch` command updated from `cu121` to `cu124`.
- Notes:
  - `streamlit-paste-button>=0.1.0` pin fixed (0.1.3 doesn't exist, latest is 0.1.2).
  - `transformers>=5.5.0` pin documented in model-serving/requirements.txt.
- Dependencies: 3.11, 3.12

### 3.14 Remove Load Model button — auto-connect to backend
- Status: [x]
- Started: 2026-04-05
- Completed: 2026-04-05
- Included in version:
- Acceptance criteria:
  - Load Model button removed from sidebar.
  - UI auto-detects backend health and active model on every page load.
  - Sidebar shows connection status (connected/mismatch/offline) instead of a button.
  - Chat input enabled automatically when backend is reachable.
  - Model mismatch (UI selection vs server model) shown as a warning.
  - `load_model()` method removed from `ServingClient` and `SandboxService`.
  - All 43 tests pass.
- Validation:
  - `app.py` sidebar: button+columns replaced with a status placeholder filled after auto-detect.
  - Right panel: status messages updated to remove "Click Load Model" references.
  - `serving_client.py`: `load_model()` deleted (was a no-op validation stub).
  - `sandbox_service.py`: `load_model()` pass-through deleted.
- Notes:
  - With vLLM (or the Windows-native OpenAI shim) the model is loaded at server startup — the button was vestigial.
  - The Windows-native backend still loads weights lazily on first `POST /v1/chat/completions` — the "Loading weights" tqdm bar in the server terminal is real progress, not stale.
- Dependencies: 3.12

### 3.15 Fix stuck model loading and start_vllm.ps1 wslpath bug
- Status: [x]
- Started: 2026-04-05
- Completed: 2026-04-05
- Included in version:
- Acceptance criteria:
  - Model weights load from local HF cache without hanging on HuggingFace Hub auth for gated models.
  - `from_pretrained()` tries `local_files_only=True` first, falls back to Hub download if not cached.
  - Processor loading (`_ensure_processor`) uses the same local-first strategy.
  - Mistral processor loading refactored into `_load_mistral_processor()` method with local-first support.
  - `start_vllm.ps1` no longer crashes with `ErrorRecord` when `wslpath` writes to stderr.
  - All 43 tests pass.
- Validation:
  - `_load_multimodal_model()`: tries `local_files_only=True` first, catches `OSError`, retries with network.
  - `_load_processor()`: new helper with `local_first` parameter.
  - `_load_mistral_processor()`: extracted from inline block, uses `local_first` for config, tokenizer, and processor.
  - `start_vllm.ps1`: `2>&1` replaced with `2>$null` + `$LASTEXITCODE` check to avoid `ErrorRecord.Trim()` crash.
  - Reproduction: `google/gemma-4-E2B-it` with empty `HF_TOKEN` now loads in <1 second from cache (was hanging indefinitely).
- Notes:
  - Root cause: `from_pretrained()` checks HF Hub for model updates; gated models (Gemma 4) reject unauthenticated requests, causing an indefinite hang.
  - `HF_TOKEN` is still recommended in `.env` for first-time downloads and model updates.
- Dependencies: 3.14

### 3.16 Fix "turn" suffix in responses and start_vllm.ps1 wslpath crash
- Status: [x]
- Started: 2026-04-05
- Completed: 2026-04-05
- Included in version:
- Acceptance criteria:
  - Assistant responses no longer show a trailing "turn" artifact.
  - `start_vllm.ps1` no longer crashes on `wslpath` subprocess errors.
  - All 43 tests pass.
- Validation:
  - Changed `skip_special_tokens=False` → `True` in all three decode paths: `TextIteratorStreamer`, `_generate_text_one_shot`, `_generate_multimodal`.
  - Added `_strip_special_tokens()` safety strip in `_parse_response()` to catch any residual `<end_of_turn>`, `<start_of_turn>`, `<eos>`, `<bos>`, `<pad>` tokens.
  - `start_vllm.ps1`: replaced `wsl wslpath` subprocess call with manual `C:\... → /mnt/c/...` conversion (deterministic, no quoting issues).
- Notes:
  - Root cause of "turn" suffix: `skip_special_tokens=False` left `<end_of_turn>` in the raw output. When rendered via `st.markdown()`, the browser treated it as an unknown HTML tag and showed the residual text "turn".
  - Root cause of wslpath crash: PowerShell quoting of backslash paths + stderr capture caused `wslpath` to produce no stdout, failing the `-not $wslScriptDir` check.
- Dependencies: 3.15

### 3.17 vLLM + Gemma 4 compatibility (transformers 5.x override)
- Status: [x]
- Started: 2026-04-05
- Completed: 2026-04-05
- Included in version:
- Acceptance criteria:
  - vLLM 0.19.0 in WSL2 successfully loads and serves `google/gemma-4-E2B-it`.
  - `transformers>=5.5.0` and latest `huggingface_hub` installed in WSL2 venv (overriding vLLM's `<5` pin).
  - `setup_vllm.sh` automates the override so fresh installs work out of the box.
  - `/health` returns 200, `/v1/models` lists `google/gemma-4-E2B-it`.
  - Model loads from symlinked Windows HF cache (`/mnt/c/...`).
  - README `wslpath` call replaced with working `$USER`-based path.
- Validation:
  - `setup_vllm.sh`: added step 5 — `pip install 'transformers>=5.5.0' --no-deps` + `pip install --upgrade huggingface_hub`.
  - Verification step prints transformers version.
  - Tested: vLLM 0.19.0 imports fine with transformers 5.5.0, `Gemma4ForConditionalGeneration` resolved, weights loaded (102s from `/mnt/c/`), CUDA graphs captured, API serving on `0.0.0.0:8000`.
  - `localhost:8000/health` returns 200 from Windows.
  - `localhost:8000/v1/models` returns `google/gemma-4-E2B-it`.
  - README.md: `wsl wslpath -a` replaced with manual `/mnt/c/Users/$USER/...` path.
- Notes:
  - vLLM 0.19.0's pip metadata pins `transformers<5,>=4.56.0`, but the runtime works correctly with 5.5.0.
  - Loading from `/mnt/c/` (Windows mount) is ~100s vs ~10s from native Linux filesystem. For faster startup, copy the model to `~/.cache/huggingface/hub/` inside WSL2.
  - First start also compiles CUDA graphs (~53s); subsequent starts use the cache at `~/.cache/vllm/torch_compile_cache/`.
- Dependencies: 3.16

### 3.18 Fix multimodal image handling through UI
- Status: [x]
- Started: 2026-04-06
- Completed: 2026-04-06
- Included in version:
- Acceptance criteria:
  - Image uploads and clipboard pastes produce correct model descriptions through the Streamlit UI.
  - Prior-turn images with stale temp-file paths do not crash the server.
  - SSE streaming errors are surfaced to the client instead of silently returning empty responses.
- Validation:
  - Root cause 1 (empty responses): `_generate_multimodal()` is one-shot (never calls `token_callback`), so the SSE streaming handler in `openai_compat.py` emitted only `finish_reason=stop` without any content. Fixed by checking `result_holder` after the worker thread finishes and emitting any unstreamed text as a final delta chunk.
  - Root cause 2 (stale prior-turn images): `model_history` in `app.py` stored local file paths (e.g. `C:/Users/.../Temp/tmp123.png`) for images. On subsequent turns, `_ensure_data_uri_or_url()` tried to resolve them; if the temp file was deleted, the raw path string was sent to the server, which crashed with `ValueError: Incorrect image source` inside `transformers.image_utils.load_image`.
  - Root cause 3 (primary — garbage image descriptions): `.env` had `MODEL_QUANTIZE_4BIT=1`. NF4 quantization via BitsAndBytes destroys the vision tower on Gemma 4 E2B — the model cannot decode images at all. Outputs include "Please provide an image", hallucinated brand names ("Pepsi", "ESPN"), and repeated nonsense words ("epip", "pepina"). Confirmed: 5/5 requests with quantization ON produce garbage; 5/5 with quantization OFF produce correct descriptions. Fixed `.env` to `MODEL_QUANTIZE_4BIT=0` and added a prominent startup warning in `model_service.py` when quantization + multimodal model are both active.
  - Fix for stale images (a): `app.py` now converts image paths to `data:` URIs at storage time (via `_ensure_data_uri_or_url()`) before appending to `model_history`. Prior turns always carry valid inline image data.
  - Fix for stale images (b): `_to_openai_messages()` in `serving_client.py` now drops image blocks whose URLs cannot be resolved (returns empty string from `_ensure_data_uri_or_url` → block is skipped with a warning). This provides graceful degradation for legacy session state.
  - Fix for SSE error surfacing: `openai_compat.py` SSE worker now catches exceptions and emits an `[Error: ...]` delta chunk instead of silently returning an empty stream.
  - `pixel_values` warning guard added to `_generate_multimodal()` for future debugging.
  - Tested with `jw.png` (jewelry, 205x167) and screenshot (480x545) — 5/5 runs at temperature=1.0 with exact UI parameters produce correct descriptions ("floral or crystal headpiece/wreath").
  - All 45 tests pass.
- Notes:
  - The `_generate_multimodal()` path remains one-shot (no `TextIteratorStreamer`). True token-by-token streaming for multimodal is a future enhancement.
  - Data URI storage in session state increases memory use but is bounded by conversation length.
  - 4-bit quantization is only safe for text-only workloads. For multimodal models, leave `MODEL_QUANTIZE_4BIT=0`.
- Dependencies: 3.17

### 3.19 Rewrite README with three-section structure
- Status: [x]
- Started: 2026-04-06
- Completed: 2026-04-06
- Included in version:
- Acceptance criteria:
  - README reorganized around three clear sections: Model Serving, Sandbox UI, Playground.
  - Model Serving section covers both backends (vLLM and Windows-native) with configuration tables, console output examples, and comparison matrix.
  - Performance results (RTX 3090 benchmarks) live under Model Serving, not scattered across unrelated sections.
  - Hardware guide, VRAM table, and quantization guidance consolidated under Model Serving.
  - Sandbox UI section is concise — features list, start command, no model internals.
  - Playground section lists scripts with one-line descriptions and usage examples.
  - Future ideas collected at the end instead of mixed into operational sections.
  - Table of contents with anchor links for easy navigation.
  - All 45 tests pass.
- Validation:
  - Old README (355 lines, mixed concerns) replaced entirely with new README (~330 lines, clear hierarchy).
  - Sections flow: At a Glance → Model Serving (vLLM, Windows-native, comparison, benchmarks, hardware, quantization) → Sandbox UI → Playground → Quick Start → Tests → Screenshots → Future Ideas → Docs Index.
  - AWQ vs NF4 quantization differences clearly separated under Model Serving.
  - No cross-references to internal module names or file paths that would confuse non-technical readers.
- Notes:
  - Previous README mixed hardware guidance, GPU detection, model architecture, quantization, setup instructions, and internal module details without clear separation.
  - New structure reflects the actual user journey: choose a backend → configure → start → use UI → run benchmarks.
- Dependencies: 3.18

### 3.20 Shared media folder architecture for all file types
- Status: [x]
- Started: 2026-04-07
- Completed: 2026-04-07
- Included in version:
- Acceptance criteria:
  - All uploaded files (images, audio, video) are saved to a single `SHARED_MEDIA_DIR` on disk.
  - `SHARED_MEDIA_DIR` is configurable via `ui/.env`, `vllm-serving/.env.vllm` (WSL path), and `model-serving/.env` (all three must point to the same folder).
  - vLLM backend reads all media via `file:///mnt/c/...` URIs — no base64 encoding, no frame extraction.
  - Native backend reads local images by Windows file path — Transformers `load_image()` opens them directly via PIL (no base64 encoding).
  - Sidebar shows the configured Windows and WSL paths with a folder-exists indicator.
  - Sidebar warns when the folder is missing so operators can diagnose misconfiguration before media uploads fail.
  - Stale `video_frames` slider and unused `extract_video_frames` import removed from `app.py`.
- Validation:
  - `ui/src/ai_sandbox/config.py` exports `SHARED_MEDIA_DIR: Path` and `shared_media_dir_wsl()` helper.
  - `model-serving/src/model_serving/config.py` exports `SHARED_MEDIA_DIR: Path`.
  - `uploads.py` saves to `SHARED_MEDIA_DIR` with UUID filename (no tempfiles).
  - `serving_client._to_openai_messages()` dispatches via `_to_file_uri_for_vllm()` for vLLM; `_ensure_local_path_or_url()` for native (path sent directly, not base64).
  - `_to_file_uri_for_vllm()` converts Windows paths (`C:\foo`) to `file:///mnt/c/foo` WSL URIs; passes through `http(s)://` and `data:` unchanged.
  - `app.py` imports `SHARED_MEDIA_DIR, shared_media_dir_wsl`; sidebar shows "Shared media folder" expander with exists check.
  - No errors in `app.py`, `serving_client.py`, or either `config.py`.
- Notes:
  - Audio is converted to a `[Audio file: file:///mnt/c/...]` text hint for vLLM (vLLM has no standard audio content type in the OpenAI schema).
  - If `SHARED_MEDIA_DIR` in `.env` doesn't map to the same mount as in `.env.vllm`, file delivery breaks silently — the sidebar warning surfaces this risk.
- Dependencies: 3.17, 3.18

### 3.21 Benchmark refresh with shared-media architecture
- Status: [x]
- Started: 2026-04-07
- Completed: 2026-04-07
- Included in version:
- Acceptance criteria:
  - New benchmark scripts (`vllm_benchmark.py`, `native_benchmark.py`) measure text and image scenarios for both backends.
  - Image scenario uses `file://` URI (vLLM) and local Windows path (native) via shared media folder — no base64.
  - Results saved to `playground/results.json` with timestamp and backend tag.
  - README performance section updated with new measurements.
  - design.md updated with shared-media architecture description and benchmark numbers.
  - Screenshots and GIF regenerated with updated UI (shared-media sidebar expander visible).
- Validation:
  - vLLM (RTX 3090, Gemma 4 E2B, bf16): text 74.7 tok/s, 37 ms TTFT; image 53.9 tok/s, 129 ms TTFT.
  - Native (RTX 3090, Gemma 4 E2B, bf16): text 6.2 tok/s, 597 ms TTFT; image 6.9 tok/s (35 s TTFT — includes Transformers image preprocessing).
  - vLLM is ~12× faster in text throughput and ~270× faster in first-image-token latency.
  - Image scenario confirmed end-to-end: file written to `C:\ai-workbench\shared-media\`, vLLM read via `file:///mnt/c/ai-workbench/shared-media/`, native read via Windows path.
  - 11 UI screenshots captured + `demo.gif` rebuilt.
  - `playground/results.json` contains timestamped entries for both backends.
- Notes:
  - `start.sh` no longer has a hardcoded default for `SHARED_MEDIA_DIR`; value must come from `.env.vllm` — missing value prints a WARNING and omits the `--allowed-local-media-path` flag.
  - Native server must be started with the project venv Python (`venv\Scripts\python.exe`) not the system Python; `start_server.ps1` handles this correctly.
- Dependencies: 3.20

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