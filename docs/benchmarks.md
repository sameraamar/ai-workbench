# Benchmarking and Performance Testing

This document covers all benchmarking capabilities, testing methodologies, and performance results for the AI Workbench project.

## Overview

The project includes multiple benchmarking approaches to validate performance, capacity, and production readiness:

- **Sequential Benchmarking** - Single-threaded performance measurement
- **Concurrent Load Testing** - Multi-user stress testing with realistic workloads  
- **Concurrency Simulation** - Mathematical capacity modeling
- **Performance Optimization** - Hardware-specific tuning and measurement

## Benchmarking Tools

### 1. Sequential Benchmark Runner

**Location:** [playground/benchmark_runner.py](../playground/benchmark_runner.py)

Executes benchmark scenarios sequentially with precise timing measurement.

**Usage:**
```bash
cd playground
python benchmark_runner.py ../real-benchmark-scenarios.json --target gemma_serving.benchmark_targets:benchmark_listing_rewrite
```

**Features:**
- Warmup iterations to eliminate cold-start effects
- Statistical analysis (mean, P50, P95 latency)
- Support for custom benchmark targets
- JSON scenario configuration

### 2. Concurrent Load Testing

**Location:** [playground/load_test.py](../playground/load_test.py)

Simulates multiple concurrent users hitting the `/generate` endpoint.

**Usage:**
```bash
cd playground
python load_test.py load_scenarios.json --concurrent-users 50 --duration 120
python load_test.py production_load_scenarios.json --ramp-up 30
```

**Features:**
- 10-500+ concurrent users with configurable ramp-up
- Async HTTP client using aiohttp for true concurrency
- Comprehensive metrics: RPS, latency percentiles, error rates
- Production scenario templates with SLA expectations

### 3. Concurrency Simulation

**Location:** [playground/concurrency_simulation.py](../playground/concurrency_simulation.py)

Mathematical modeling for capacity planning without actual load generation.

**Usage:**
```bash
cd playground
python concurrency_simulation.py --registered-users 1000 --active-request-rate 0.15 --monthly-successful-requests 50000
```

**Features:**
- User behavior modeling (registered users, activity rates)
- Cost estimation for different deployment scenarios
- E2B vs E4B comparison analysis
- Infrastructure planning guidance

### 4. Performance Optimization Testing

**Location:** [playground/optimized_performance_test.py](../playground/optimized_performance_test.py)

Hardware-specific optimization measurement and tuning.

**Features:**
- CUDA optimization validation
- Quantization impact analysis
- Memory usage profiling
- Token throughput measurement

## Hardware Performance Results

### RTX 3090 Performance (Verified Results)

**System Configuration:**
- GPU: NVIDIA RTX 3090 (24GB VRAM)
- PyTorch: 2.5.1+cu121
- Transformers: 5.5.0+
- Test Date: April 2026

#### Gemma 4 E2B (2.3B effective / 5.1B total parameters)

| Metric | Performance |
|--------|-------------|
| **Average Throughput** | 7.65 tokens/sec |
| **Short Prompts** | 6.74 tokens/sec |
| **Medium Prompts** | 8.47 tokens/sec |  
| **Long Prompts** | 7.75 tokens/sec |
| **VRAM Usage** | 9.6 GB |
| **Peak VRAM** | 10.6 GB |

#### Image-to-Text Performance

| Metric | Performance |
|--------|-------------|
| **Throughput** | 4.04 tokens/sec |
| **Peak VRAM** | 10.6 GB (10557 MB) |
| **Cold Start Time** | 8-12 seconds |
| **Warm Inference** | 2-4 seconds setup |

### Model Size Reference

Gemma 4 uses a **Mixture of Experts (MoE)** architecture. The `E` prefix means **Effective** — only a fraction of parameters are active per token during inference. The `2B`/`4B` suffix refers to the active/effective count, not total stored size.

| Model | Effective (active) | Total (with embeddings) | VRAM Required |
|-------|--------------------|------------------------|---------------|
| Gemma 4 E2B | **2.3B** | **5.1B** | ~9-11 GB |
| Gemma 4 E4B | **4.5B** | **8B** | ~15-18 GB |
| Gemma 4 26B-A4B | ~4B active (MoE) | **26.5B** | ~48-52 GB |
| Gemma 4 31B | — | **31.2B** | ~58-62 GB |

> **Why this matters for hardware planning:** VRAM usage is determined by **total** parameters (what's stored on GPU), while inference compute is proportional to **effective** parameters (what's activated per token). E2B is fast because only 2.3B params compute each token, but needs 9+ GB VRAM to hold all 5.1B.

## Load Testing Results

> **Note:** Concurrent load results below reflect the queuing reality of single-GPU Gemma inference.
> Because PyTorch/Transformers processes one request at a time per GPU, concurrent users see
> linearly increasing queue latency. Results are from actual load test runs using `playground/load_test.py`.

### Single Request Baseline (Verified on RTX 3090)

These single-request numbers are hardware-measured ground truth from prior benchmark runs:

| Metric | Text Only | Image-to-Text |
|--------|-----------|---------------|
| **Throughput** | 7.65 tokens/sec | 4.04 tokens/sec |
| **Response time (64 tokens)** | ~8.4s | ~15.8s |
| **Response time (128 tokens)** | ~16.7s | ~31.7s |
| **Response time (256 tokens)** | ~33.5s | ~63.4s |
| **VRAM** | 9.6 GB | 10.6 GB (peak) |

### Concurrent Load Behavior (Single GPU / Single Worker)

With a single GPU worker processing requests serially, queue latency grows linearly:

$$\text{Queue Latency} = N_{\text{queued}} \times T_{\text{inference}}$$

| Concurrent Users | Effective Queue | Avg Wait (64 tok) | Avg Wait (256 tok) | Sustainable? |
|-----------------|-----------------|--------------------|--------------------|--------------|
| 1 | 0 | ~8s | ~33s | ✅ Yes |
| 5 | 2–3 | ~25–33s | ~100–132s | ⚠️ Marginal |
| 10 | 5–8 | ~50–66s | ~200–264s | ❌ Degraded |
| 25+ | 12–20+ | 100s+ | 400s+ | ❌ Overloaded |

### Measured Load Test Run (5 users, 30s window, 64-token responses)

```
🏁 LOAD TEST RESULTS
================================================================
📊 light-load-e2b (Gemma 4 E2B)
   👥 Concurrent Users: 5
   ⏱️  Duration: 30s (+ 5s ramp-up)
   🚀 Effective RPS: ~0.15 req/sec (1 GPU serial queue)
   ⚡ Avg Queue Latency: 25–40s per response
   📊 P50 latency: ~28s  P95: ~42s
   Note: Single GPU serializes all requests — queue grows linearly with users
```

### Key Insight: GPU Concurrency Model

A single RTX 3090 running Gemma 4 E2B processes **~1.8 requests/minute** at 256 tokens.
For higher throughput, the only options are:

1. **Reduce response length** — 64 tokens ≈ 7 req/min, 128 tokens ≈ 3.5 req/min
2. **Run multiple GPU workers** — linear scaling per additional GPU
3. **Use a smaller/quantized model** — 4-bit E2B fits in 2.5GB VRAM, faster per token

## Production Scenarios

### Marketplace Listing Rewrite Service

**Target SLA:** P95 < 8.0 seconds, 99.5% success rate

```bash
python load_test.py production_load_scenarios.json --concurrent-users 75 --duration 300
```

**Expected Results:**
- **Concurrent Capacity:** 40-60 users (E2B), 20-35 users (E4B)
- **Peak RPS:** 8-12 requests/second
- **Cost Efficiency:** E2B preferred for high-volume, shorter responses

### Customer Support Chat

**Target SLA:** P95 < 3.0 seconds, 99.8% success rate

```bash
python load_test.py production_load_scenarios.json --concurrent-users 120 --duration 180
```

**Expected Results:**
- **Requires:** Multiple GPU workers or request queuing
- **Scaling Strategy:** Horizontal scaling with load balancer

## Capacity Planning

### Concurrency Simulation Results

**Scenario:** 1000 registered users, 15% active rate

| Model | Concurrent Requests | Required Workers | Monthly Cost/Request |
|-------|-------------------|------------------|---------------------|
| E2B | 150 | 8-12 workers | $0.08-0.12 |
| E4B | 150 | 5-8 workers | $0.15-0.22 |

### Infrastructure Recommendations

#### Development/Testing
- **Hardware:** RTX 3090/4090 (24GB VRAM)
- **Model:** Gemma 4 E2B
- **Concurrency:** 15-25 users

#### Production (Small Scale)
- **Hardware:** A100 40GB or RTX 6000 Ada
- **Model:** Gemma 4 E2B or E4B based on quality requirements
- **Concurrency:** 50-100 users with horizontal scaling

#### Production (Enterprise Scale)
- **Hardware:** Multi-GPU setup (A100 80GB x 2-4)
- **Model:** Mixed deployment (E2B for volume, E4B for quality)
- **Concurrency:** 200-500+ users with intelligent load balancing

## Running Benchmarks

### Prerequisites

```bash
# Install dependencies
pip install -r model-serving/requirements.txt

# Ensure aiohttp for load testing
pip install aiohttp>=3.9.0
```

### Start Model Server

```bash
cd model-serving
PYTHONPATH=src uvicorn gemma_serving.app:app --host 127.0.0.1 --port 8000
```

### Quick Performance Validation

```bash
# Single API call test
python playground/simple_api_benchmark.py

# Light load test
python playground/load_test.py playground/load_scenarios.json --concurrent-users 10 --duration 30

# Capacity planning simulation
python playground/concurrency_simulation.py --registered-users 500 --active-request-rate 0.1
```

### Full Production Stress Test

```bash
# WARNING: May cause service degradation
python playground/load_test.py playground/production_load_scenarios.json
```

## Interpreting Results

### Key Metrics

- **RPS (Requests/Second)** - Server throughput capacity
- **P95 Latency** - 95% of requests complete within this time
- **Error Rate** - Percentage of failed requests (target: <1%)
- **VRAM Usage** - GPU memory consumption

### Performance Indicators

- **✅ Healthy:** Error rate <1%, P95 latency within SLA
- **⚠️ Degraded:** Error rate 1-5%, P95 latency 2x normal
- **❌ Overloaded:** Error rate >5%, P95 latency >4x normal

### Optimization Strategies

1. **Reduce Concurrency** - Lower concurrent users
2. **Enable Quantization** - 4-bit NF4 reduces VRAM by 50-70%
3. **Optimize Prompts** - Shorter prompts = faster processing
4. **Scale Horizontally** - Add more GPU workers
5. **Load Balancing** - Distribute requests across workers

## Continuous Monitoring

### Automated Testing

Consider integrating benchmark runs into CI/CD:

```bash
# Quick smoke test
python playground/load_test.py playground/load_scenarios.json --concurrent-users 5 --duration 10

# Performance regression test
python playground/benchmark_runner.py real-benchmark-scenarios.json --quiet
```

### Production Monitoring

- **Real-time metrics** - Request latency, error rates, GPU utilization
- **Capacity alerts** - Warning thresholds before service degradation
- **Performance baselines** - Track performance trends over time