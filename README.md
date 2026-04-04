# Gemma Sandbox Arena

A local-first sandbox for exploring multimodal AI models through text, image, audio, and video understanding workflows. Currently built around Google Gemma 4, with a serving research toolkit for benchmarking and capacity planning.

The UI and model-serving backend communicate through a generic `POST /generate` API, so swapping or adding models (Llama, Phi, Mistral, etc.) only requires changes inside `model-serving/`.

## Repository Layout

| Folder | Purpose | Entry point |
|---|---|---|
| `model-serving/` | FastAPI model-serving backend (loads Gemma, exposes `/generate` and job endpoints) | `uvicorn gemma_serving.app:app` |
| `ui/` | Streamlit sandbox UI (calls model-serving over HTTP) | `streamlit run ui/app.py` |
| `playground/` | Standalone demo and benchmark scripts | Individual `.py` files |

Each project has its own `requirements.txt`, `.env.example`, and `PYTHONPATH` root.

## Hardware Recommendations

Gemma 4 inference performance depends heavily on having a CUDA-capable GPU. CPU-only inference is functional but too slow for interactive or production use.

### Measured CPU-Only Baseline

These results were collected on a Windows machine with no CUDA GPU, using default FP32 weights and `max_new_tokens=192`:

| Model | Task | Latency per request |
|---|---|---|
| Gemma 4 E2B | Text-only listing rewrite | ~35–43 seconds |
| Gemma 4 E4B | Text-only listing rewrite | ~77–101 seconds |

**Verdict:** CPU-only inference is usable for offline batch work but not for interactive or multi-user serving.

### Recommended GPU Tiers

#### Gemma 4 E2B (smallest, lowest cost)

| GPU | VRAM | Expected text-rewrite latency | Notes |
|---|---|---|---|
| NVIDIA RTX 3060 12 GB | 12 GB | 2–5 seconds | Budget entry point, fits E2B in FP16 |
| NVIDIA RTX 4060 Ti 16 GB | 16 GB | 1.5–4 seconds | More headroom for longer prompts |
| NVIDIA T4 (cloud) | 16 GB | 2–5 seconds | Free tier on Colab, cheapest cloud option |

#### Gemma 4 E4B (recommended default)

| GPU | VRAM | Expected text-rewrite latency | Notes |
|---|---|---|---|
| NVIDIA RTX 4070 Ti 12 GB | 12 GB | 3–6 seconds | May need 8-bit quantization to fit |
| NVIDIA RTX 3090 / 4080 | 16–24 GB | 2–5 seconds | Comfortable fit in FP16 |
| NVIDIA A10G (cloud) | 24 GB | 2–4 seconds | Good cloud option, ~$0.75/hr spot |
| NVIDIA L4 (cloud) | 24 GB | 2–4 seconds | Available on GCP, efficient inference card |

#### Gemma 4 26B A4B / 31B (not recommended for low-cost)

These models require 40–80 GB VRAM (A100, H100 class) and are not practical for local or budget deployments.

### Cost-Effective Strategies

1. **Start with E2B on a 12–16 GB GPU.** Validate rewrite quality before investing in larger hardware.
2. **Use quantization (8-bit or 4-bit)** to fit larger models on smaller GPUs, at a small quality cost.
3. **Use cloud GPU spot instances** (Colab T4 free tier, Lambda Labs, RunPod, Vast.ai) for experimentation before buying hardware.
4. **Reduce `max_new_tokens`** to lower latency. Many listing rewrites complete well under 192 tokens.
5. **Keep image analysis asynchronous.** Multimodal requests are 2–3× slower than text-only; queue them as background jobs.
6. **Cache repeated rewrites.** The FastAPI blueprint includes an in-memory cache to avoid re-running identical requests.

### Minimum Hardware Summary

| Deployment Goal | Minimum GPU | Minimum VRAM | Model |
|---|---|---|---|
| Local prototyping | RTX 3060 | 12 GB | E2B |
| Local interactive use | RTX 3090 / 4080 | 16–24 GB | E4B |
| Cloud serving (low cost) | T4 / L4 | 16–24 GB | E2B or E4B |
| Production 100-user serving | Multiple L4 / A10G workers | 24 GB each | E4B |

## Quick Start

```bash
# Clone and create a virtual environment
python -m venv venv
venv\Scripts\Activate.ps1   # Windows PowerShell
# source venv/bin/activate  # Linux/macOS

# Install dependencies for both projects
pip install -r model-serving/requirements.txt
pip install -r ui/requirements.txt

# Copy and configure environment for each project
cp model-serving/.env.example model-serving/.env
# Edit model-serving/.env to set GEMMA_MODEL_ID and GEMMA_FASTAPI_GATEWAY
cp ui/.env.example ui/.env
# Edit ui/.env to set SERVING_URL if not using default http://localhost:8000

# Start the model-serving API (terminal 1)
cd model-serving
$env:PYTHONPATH = "src"          # Windows PowerShell
# export PYTHONPATH=src          # Linux/macOS
uvicorn gemma_serving.app:app --host 127.0.0.1 --port 8000

# Start the UI (terminal 2)
cd ui
$env:PYTHONPATH = "src"          # Windows PowerShell
# export PYTHONPATH=src          # Linux/macOS
streamlit run app.py
```

## Serving Research Toolkit

The `playground/` directory contains standalone tools for benchmarking and capacity planning:

```bash
# Run simulated benchmark harness validation
python playground/benchmark_runner.py model-serving/tests/scenarios.json

# Run real Gemma inference benchmarks
$env:PYTHONPATH = "model-serving\src"   # Windows PowerShell
python playground/benchmark_runner.py model-serving/docs/scenarios/ebay-listing-benchmarks.json \
  --target gemma_serving.benchmark_targets:benchmark_listing_rewrite

# Run E2B vs E4B concurrency simulation
python playground/concurrency_simulation.py --registered-users 100 --active-request-rate 0.1 --multimodal-share 0.2
```

## Documentation

- [docs/START_HERE.md](docs/START_HERE.md) — Project entrypoint and restart guide
- [docs/tasks.md](docs/tasks.md) — Task tracking and phase status
- [docs/design/design.md](docs/design/design.md) — Architecture and design decisions
- [docs/research/gemma4-serving-evaluation.md](docs/research/gemma4-serving-evaluation.md) — Model selection and serving research
- [docs/research/low-cost-fastapi-blueprint.md](docs/research/low-cost-fastapi-blueprint.md) — Queue-first FastAPI blueprint design

## License

See repository for license details.
