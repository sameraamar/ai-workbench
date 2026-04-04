# Purpose

This document describes a concrete low-cost FastAPI serving blueprint for the Gemma serving research package.
It exists so future implementation work can start from a minimal, defensible serving shape rather than redesigning the same constraints repeatedly.
AI agents and humans should update this document when the blueprint changes or measured results invalidate its assumptions.

# Maintenance Instructions

Update this file when the serving topology, API surface, caching strategy, or queueing assumptions change.
Humans or AI may update it.
Keep this aligned with the code in `src/gemma_serving_research/low_cost_fastapi.py`.
The real Gemma gateway implementation lives in `src/gemma_serving_research/gemma_gateway.py`.

## Scope

This blueprint targets the cheapest practical FastAPI shape for Gemma-backed listing rewrite and optional image attribute extraction.

It deliberately favors:

- one process
- one in-memory worker queue
- one warm model runtime behind the worker
- in-memory result caching
- asynchronous job polling instead of synchronous long-held HTTP requests

It deliberately does not try to solve:

- multi-host coordination
- durable queueing
- durable job storage
- autoscaling
- enterprise observability

## Code Location

- FastAPI blueprint: `src/gemma_serving_research/low_cost_fastapi.py`
- Real Gemma gateway: `src/gemma_serving_research/gemma_gateway.py`
- Planning helpers: `src/gemma_serving_research/planning.py`
- Concurrency simulation: `src/gemma_serving_research/concurrency_simulation.py`
- Benchmark harness: `src/gemma_serving_research/benchmark_runner.py`

## Why This Shape Fits Low Cost

- One warm worker avoids repeated cold loads.
- Queueing prevents the API process from trying to run too many model calls at once.
- Job polling is cheaper and operationally safer than keeping many slow requests open.
- In-memory cache can remove duplicate work for repeated product rewrites.
- The same structure can later be upgraded to Redis, Celery, or multiple workers without changing the request contract.

## Endpoints

### `GET /health`

Returns process health, queue depth, and whether cache is enabled.

### `POST /jobs/rewrite`

Accepts a listing rewrite request with title and description.
Returns a job id immediately.

### `POST /jobs/extract-attributes`

Accepts one or two image URLs plus optional attribute hints.
Returns a job id immediately.

### `GET /jobs/{job_id}`

Returns the current job state:

- `pending`
- `running`
- `succeeded`
- `failed`

## Request Path

1. Client submits a rewrite or attribute extraction job.
2. API normalizes and hashes the payload.
3. Cached results return immediately when available.
4. New jobs enter a bounded in-memory queue.
5. Single worker processes the queue with a gateway implementation.
6. Client polls for job completion.

## Recommended First Deployment

- Model: `google/gemma-4-E2B-it`
- Worker count: 1
- Queue size: 100 or lower depending on memory budget
- Request style: text-only rewrite by default
- Image analysis: optional second call, limited to 1 or 2 images
- Cache: enabled

## Upgrade Path

When the low-cost blueprint stops being sufficient, upgrade in this order:

1. Replace the stub gateway with a real Gemma-backed gateway.
2. Move from in-memory cache to Redis if repeated work becomes important across restarts.
3. Move from in-memory queue to a durable queue if job loss on restart becomes unacceptable.
4. Add more workers or a second host only after benchmark evidence shows queue delay is the actual bottleneck.
5. Consider E4B only if E2B rewrite quality is not acceptable.

## Run Command

Example local run command:

```bash
uvicorn gemma_serving_research.low_cost_fastapi:app --host 127.0.0.1 --port 8000
```

## Next Code Step

The blueprint currently uses a stub gateway for safe local verification.
The next integration step is to add a real gateway that wraps Gemma inference while preserving the same API contract.

That real gateway now exists and can be enabled by setting `GEMMA_FASTAPI_GATEWAY=gemma` in `.env` or the process environment.
The stub path remains the default because it is safer for local smoke tests and documentation demos.