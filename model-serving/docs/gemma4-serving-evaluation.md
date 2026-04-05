# Purpose

This document tracks model-selection and serving research for productionizing Gemma 4 workloads related to ecommerce listing generation.
It exists so future implementation and infrastructure choices can be driven by recorded assumptions, benchmarks, and cost analysis rather than chat history.
AI agents and humans should update this document as measured results replace planning estimates.

# Maintenance Instructions

Update this file when benchmark results, throughput assumptions, infrastructure choices, or cost estimates change.
Humans or AI may update it.
Keep a clear distinction between measured numbers and planning estimates.

## Research Goal

Choose the most appropriate Gemma 4 model and serving topology for a FastAPI system that:

- rewrites product titles and descriptions for marketplace listings such as eBay
- optionally inspects one or two product images to extract attributes
- may need to serve up to 100 users, with unknown but potentially meaningful concurrency
- should prefer free local operation or the lowest practical operating cost

## Workload Summary

- Primary path: text-only rewrite of title and description
- Secondary path: multimodal attribute extraction from one or two images plus short text instructions
- Possible source media volume: up to 24 images per product, but not all images should be sent to the model in a single request
- Expected serving surface: FastAPI-backed web system
- Multi-user requirement: plan for bursty concurrent usage rather than assuming one-at-a-time operator workflows

## Terms

### Multimodal request

A multimodal request is a model call that includes more than one modality in the prompt, for example:

- image plus text
- audio plus text
- video frames plus text

For this project, the practical multimodal case is image-plus-text attribute extraction.

### HF Hub authentication warning

When Transformers downloads model files from Hugging Face without an `HF_TOKEN`, requests are anonymous.
Anonymous requests still work for public models, but they typically get:

- lower rate limits
- slower or less reliable downloads during contention
- less headroom for repeated worker restarts or parallel cold starts

Implication for this project:

- `HF_TOKEN` does not make inference faster after the model is already loaded in memory
- `HF_TOKEN` does help model download reliability, cold-start preparation, and container or worker bootstrap flows

After the first successful download on the same machine and under the same model cache location, later runs usually reuse the local cache and avoid redownloading model files.
That means download time is often near zero on subsequent warm starts on the same host.

Download time comes back when one of these is true:

- a new machine or container starts with an empty cache
- the cache directory is deleted or not persisted
- a different model revision is requested
- multiple workers need to populate their caches independently
- Hugging Face metadata or missing artifacts still need to be checked or fetched

## Source Inputs

- Raw Gemma 4 source copy: [docs/research/sources/gemma4-model-card.md](./sources/gemma4-model-card.md)
- Current sandbox research summary: [docs/research/research-notes.md](./research-notes.md)
- Reusable planning helpers for serving analysis live in `src/model_serving/planning`.
- Benchmark harness lives in `src/model_serving/planning/benchmarking.py`.
- Real benchmark targets for actual inference runs live in `src/model_serving/planning/benchmark_targets.py`.
- E2B versus E4B concurrency simulation lives in `src/model_serving/planning/simulation.py`.
- Low-cost FastAPI blueprint lives in [docs/research/low-cost-fastapi-blueprint.md](./low-cost-fastapi-blueprint.md) and `src/model_serving/low_cost_fastapi.py`.
- A real model-backed gateway for that blueprint now lives in `src/model_serving/gateway.py` and can be enabled with `MODEL_GATEWAY=model`.
- Ready-to-run realistic rewrite scenarios live in [docs/research/scenarios/ebay-listing-benchmarks.json](./scenarios/ebay-listing-benchmarks.json).

## Gemma 4 Candidate Models

| Model | Strengths | Weaknesses | Fit For This Work |
|---|---|---|---|
| Gemma 4 E2B | Lowest serving cost, smallest hardware target, supports image and audio | Lower quality ceiling, less headroom for nuanced rewriting | Viable fallback when hardware budget dominates |
| Gemma 4 E4B | Best quality-to-latency balance for text rewrite plus light image understanding | Still requires real serving infrastructure for concurrency | Recommended default starting point |
| Gemma 4 26B A4B | Higher quality and stronger reasoning, MoE reduces active compute vs total size | More expensive to host, lower practical concurrency per worker | Candidate if E4B quality is insufficient |
| Gemma 4 31B | Highest quality ceiling in family | Heaviest and least practical serving option for this workload | Not recommended as initial production choice |

## Initial Recommendation

Use `google/gemma-4-E4B-it` as the primary model for the first production serving design.

Reasoning:

- The core business workload is structured text rewriting, not frontier reasoning.
- The image use case is selective and should usually involve one or two images, not the full product gallery.
- E4B keeps native image understanding while remaining materially easier to serve than 26B or 31B.
- E2B is attractive for budget-sensitive deployments, but E4B is more likely to meet copy quality expectations for customer-facing listing generation.

## Low-Cost Recommendation

If free or very low cost is the dominant requirement, start with `google/gemma-4-E2B-it` on one local machine and validate whether rewrite quality is good enough.

Decision rule:

- choose E2B first when minimizing infrastructure cost matters more than maximizing rewrite quality
- move to E4B only if E2B quality is not acceptable for listing titles, descriptions, or image attribute extraction
- avoid 26B A4B and 31B for an initial low-cost deployment

Important tradeoff:

- free local serving is realistic for prototyping or low traffic
- free serving for a real 100-user production system is not realistic unless concurrency is very low and you already own the hardware

## Prompt And Request Shaping Guidance

- Keep title rewrite and description rewrite on the primary text-only path whenever possible.
- Treat image analysis as a secondary enrichment call.
- Do not send all 24 images for a product unless a later benchmark proves it is necessary.
- For multimodal prompts, put images before text, following the Gemma 4 model-card guidance.
- Keep output contracts structured, ideally JSON for attribute extraction and constrained text fields for listing output.

## Response Time Planning Estimates

The following are planning estimates, not measured benchmarks.
Actual latency depends on GPU class, quantization, prompt size, output length, batching, framework choice, and whether images are attached.

| Model | Request Type | Warm Latency Planning Range |
|---|---|---|
| E2B | text-only rewrite | 1.5s to 4s |
| E2B | image plus text | 3s to 8s |
| E4B | text-only rewrite | 2s to 6s |
| E4B | image plus text | 4s to 12s |
| 26B A4B | text-only rewrite | 4s to 12s |
| 26B A4B | image plus text | 7s to 20s |

Cold starts are expected to be much slower because model weights and processor assets may need to be downloaded and loaded.
Production serving should assume warm workers, not user-visible cold starts.

## Concurrency Planning

The phrase "100 users" is not enough to size infrastructure. The real variable is concurrent active requests.

### Planning formula

If one worker effectively handles one request at a time, then:

$$
\text{worker throughput} \approx \frac{1}{\text{average request latency in seconds}}
$$

and:

$$
\text{required workers} \approx \text{target concurrent requests}
$$

for a first-order planning model without aggressive batching.

### Example concurrency scenarios

| Registered Users | Estimated Active Request Rate | Concurrent Requests To Plan For |
|---|---|---|
| 100 | 5% | 5 |
| 100 | 10% | 10 |
| 100 | 20% | 20 |

Interpretation:

- 100 total users is easy if only a few are active concurrently.
- 100 simultaneous requests is a true serving-scale problem and should not be assumed to fit on one worker.
- Text-only traffic can often be served faster and more densely than image-assisted traffic.

## Architecture Direction

Recommended first production layout:

- FastAPI for API surface, auth, validation, and orchestration
- separate inference workers from the web API process
- queue-backed workload control for burst handling
- warm worker pool to avoid user-facing cold starts
- separate text-only and multimodal routes or job types
- caching for repeated product rewrites and repeated image-attribute extraction

Avoid coupling all traffic to one generic endpoint with one generic prompt.

## Benchmark Plan

Benchmarks should be measured for at least these scenarios:

1. Text-only title rewrite
2. Text-only title plus description rewrite
3. One-image attribute extraction
4. Two-image attribute extraction
5. Mixed burst load with 80% text-only and 20% multimodal

For each scenario, record:

- model name
- quantization mode
- hardware SKU
- average input token count
- average output token count
- p50 latency
- p95 latency
- requests per minute at steady state
- error rate
- GPU memory usage

### Current runnable paths

- Simulated harness validation: `python playground/benchmark_runner.py model-serving/tests/scenarios.json`
- Real text-rewrite benchmark target: `python playground/benchmark_runner.py model-serving/docs/scenarios/ebay-listing-benchmarks.json --target model_serving.planning.benchmark_targets:benchmark_listing_rewrite`
- Quiet mode when you only want final JSON: add `--quiet`

The simulated path validates the harness only.
The real target path invokes actual Gemma inference through the low-cost gateway.
By default the runner now emits progress logs for scenario loading, warmup, measured iterations, and Gemma runtime stages so long-running live benchmarks are easier to observe.

## Cost Of Goods Model

Use this first-pass formula for per-request infrastructure cost:

$$
\text{cost per request} = \frac{\text{monthly GPU cost} + \text{monthly infra overhead}}{\text{monthly successful requests}}
$$

Where monthly infra overhead includes at minimum:

- API hosting
- queue and cache infrastructure
- storage for input images and logs
- monitoring

To compare models fairly, estimate cost separately for:

- text-only requests
- multimodal requests
- blended production traffic

## Open Questions

- What GPU class is available for the first production deployment?
- Is quantized inference acceptable for listing quality?
- What is the expected average output length for title and description rewrites?
- Will image extraction be synchronous in the user request path or asynchronous background enrichment?
- How many truly concurrent requests should the system sustain at p95 under 10 seconds?

## Next Experiments

1. Benchmark E2B and E4B on representative listing rewrite prompts.
2. Measure one-image and two-image attribute extraction latency.
3. Simulate concurrent traffic with a mixed request profile.
4. Compute monthly cost per successful request for the smallest acceptable worker pool.
5. Escalate to 26B A4B only if measured rewrite quality or extraction accuracy is not acceptable on E4B.
