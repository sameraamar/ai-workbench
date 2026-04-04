# Playground

Standalone experiments and tools that aren't part of either deployable project.

## Scripts

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

They also depend on `torch` and `transformers` (for model inference demos) and optionally the `gemma_serving` package from `model-serving/`.
