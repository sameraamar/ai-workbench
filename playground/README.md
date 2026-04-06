# Playground

Standalone experiments and tools that aren't part of either deployable project.

## Scripts

### vLLM experiments (WSL2 / Linux)

- **vllm_gemma4.py** — Tests Gemma 4 via vLLM in both client mode (talks to a running server) and offline mode (loads model in-process). Covers text, multimodal, and streaming.
- **vllm_mistral.py** — Tests Mistral Small 3.1 24B via vLLM. RTX 3090 path uses the AWQ 4-bit quant (~14 GB VRAM). Full-precision path documented for multi-GPU setups.

Quick start (inside WSL2):
```bash
pip install vllm --upgrade

# Gemma 4 E2B (fits on RTX 3090 at bf16)
vllm serve google/gemma-4-e2b-it --host 0.0.0.0 --port 8000 --max-model-len 8192

# Mistral Small 3.1 24B — AWQ 4-bit (fits on RTX 3090)
vllm serve mistralai/Mistral-Small-3.1-24B-Instruct-2503 \\
  --quantization awq --host 0.0.0.0 --port 8000
```

Then from Windows or WSL2:
```bash
python playground/vllm_gemma4.py
python playground/vllm_mistral.py
```

### Transformers / legacy

- **gemma4_text_demo.py** — Direct Hugging Face Gemma 4 text inference demo. No project dependencies. Run with any venv that has `torch` and `transformers`.
- **benchmark_runner.py** — CLI harness for running benchmark scenarios against the model-serving API.
- **concurrency_simulation.py** — E2B vs E4B serving capacity simulation.
- **load_test.py** — Concurrent load testing tool for stress testing the `/generate` endpoint with multiple concurrent users.
- **simple_api_benchmark.py** — Single request timing test for basic API validation.

## Load Testing

The **load_test.py** script provides comprehensive concurrent load testing capabilities:

### Basic Usage

```bash
# Light load test (10 concurrent users for 60 seconds)
python load_test.py load_scenarios.json

# Custom concurrent users and duration 
python load_test.py load_scenarios.json --concurrent-users 50 --duration 120

# Production stress testing
python load_test.py production_load_scenarios.json --concurrent-users 100 --ramp-up 30
```

### Scenario Files

- **load_scenarios.json** — Development load tests with 10-100 concurrent users
- **production_load_scenarios.json** — Production-realistic scenarios with SLA expectations

### Load Test Metrics

The tool provides comprehensive metrics:
- **Requests per second** (throughput)
- **Latency percentiles** (P50, P95, P99) 
- **Success/error rates**
- **Concurrent user simulation**
- **Gradual ramp-up** to avoid overwhelming the server
- **Resource usage** tracking

### Example Output

```
🏁 LOAD TEST RESULTS
================================================================================

📊 medium-load-e2b (Gemma 4 E2B)
------------------------------------------------------------
   👥 Concurrent Users: 25
   ⏱️  Duration: 120s
   📈 Total Requests: 245
   ✅ Success Rate: 98.4%
   🚀 Requests/sec: 2.04
   ⚡ Avg Latency: 4.126s
   📊 P50/P95/P99: 3.892s / 7.234s / 9.156s
   📦 Total Data: 156.7 KB
```

### Requirements

Load testing requires `aiohttp` for async HTTP requests:

```bash
pip install -r playground/requirements.txt
```

## Dependencies

These scripts use whichever Python environment is active. Install dependencies with:

```bash
pip install -r playground/requirements.txt
```

- The **Transformers demos** need `torch` and `transformers`.
- The **vLLM scripts** need `vllm` (Linux/WSL2 only) and `requests`.
- The **load testing scripts** need `aiohttp`.
- The legacy scripts optionally depend on the `model_serving` package from `model-serving/`.
