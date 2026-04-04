from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from queue import Empty, Queue
from threading import Lock, Thread
from typing import Any, Protocol
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class JobState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ListingRewriteRequest(BaseModel):
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    marketplace: str = Field(default="ebay")
    category_hint: str | None = None


class AttributeExtractionRequest(BaseModel):
    image_urls: list[str] = Field(min_length=1)
    attribute_hints: list[str] = Field(default_factory=list)
    max_images: int = Field(default=2, ge=1, le=2)


class GenerateRequest(BaseModel):
    messages: list[dict[str, Any]]
    model_id: str | None = None
    max_new_tokens: int = Field(default=256, ge=1, le=8192)
    temperature: float = Field(default=1.0, ge=0.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    top_k: int = Field(default=64, ge=1)
    enable_thinking: bool = False
    stream_output: bool = False


class GenerateResponse(BaseModel):
    text: str
    input_token_count: int | None = None
    output_token_count: int | None = None
    total_token_count: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LoadModelRequest(BaseModel):
    model_id: str = Field(min_length=1)


class LoadModelResponse(BaseModel):
    model_id: str
    status: str
    message: str


class JobAcceptedResponse(BaseModel):
    job_id: str
    status: JobState
    cached: bool = False


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobState
    result: dict[str, Any] | None = None
    error: str | None = None


class InferenceGateway(Protocol):
    def rewrite_listing(self, request: ListingRewriteRequest) -> dict[str, Any]: ...

    def extract_attributes(self, request: AttributeExtractionRequest) -> dict[str, Any]: ...


class StubLowCostGateway:
    def rewrite_listing(self, request: ListingRewriteRequest) -> dict[str, Any]:
        rewritten_title = f"eBay: {request.title.strip()}"
        rewritten_description = request.description.strip()
        return {
            "title": rewritten_title[:80],
            "description": rewritten_description,
            "marketplace": request.marketplace,
            "category_hint": request.category_hint,
        }

    def extract_attributes(self, request: AttributeExtractionRequest) -> dict[str, Any]:
        inspected_images = request.image_urls[: request.max_images]
        return {
            "inspected_images": inspected_images,
            "attribute_hints": request.attribute_hints,
            "suggested_attributes": {
                "image_count": len(inspected_images),
            },
        }


@dataclass(frozen=True)
class LowCostServingConfig:
    queue_max_size: int = 100
    enable_cache: bool = True


@dataclass
class _QueuedJob:
    job_id: str
    kind: str
    payload: ListingRewriteRequest | AttributeExtractionRequest
    cache_key: str


class _JobRuntime:
    def __init__(self, gateway: InferenceGateway, config: LowCostServingConfig) -> None:
        self._gateway = gateway
        self._config = config
        self._jobs: dict[str, JobStatusResponse] = {}
        self._queue: Queue[_QueuedJob | None] = Queue(maxsize=config.queue_max_size)
        self._cache: dict[str, dict[str, Any]] = {}
        self._lock = Lock()
        self._worker: Thread | None = None

    def start(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = Thread(target=self._run, daemon=True)
        self._worker.start()

    def stop(self) -> None:
        self._queue.put(None)
        if self._worker is not None:
            self._worker.join(timeout=2)

    def enqueue_rewrite(self, payload: ListingRewriteRequest) -> JobAcceptedResponse:
        return self._enqueue("rewrite", payload)

    def enqueue_attribute_extraction(self, payload: AttributeExtractionRequest) -> JobAcceptedResponse:
        return self._enqueue("extract-attributes", payload)

    def get_job(self, job_id: str) -> JobStatusResponse:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            return job

    def queue_size(self) -> int:
        return self._queue.qsize()

    def gateway_name(self) -> str:
        return self._gateway.__class__.__name__

    def _enqueue(
        self,
        kind: str,
        payload: ListingRewriteRequest | AttributeExtractionRequest,
    ) -> JobAcceptedResponse:
        cache_key = _build_cache_key(kind, payload.model_dump())
        if self._config.enable_cache:
            cached_result = self._cache.get(cache_key)
            if cached_result is not None:
                job_id = str(uuid4())
                response = JobStatusResponse(
                    job_id=job_id,
                    status=JobState.SUCCEEDED,
                    result=cached_result,
                )
                with self._lock:
                    self._jobs[job_id] = response
                return JobAcceptedResponse(job_id=job_id, status=JobState.SUCCEEDED, cached=True)

        if self._queue.full():
            raise HTTPException(status_code=429, detail="Worker queue is full")

        job_id = str(uuid4())
        queued = _QueuedJob(job_id=job_id, kind=kind, payload=payload, cache_key=cache_key)
        with self._lock:
            self._jobs[job_id] = JobStatusResponse(job_id=job_id, status=JobState.PENDING)
        self._queue.put(queued)
        return JobAcceptedResponse(job_id=job_id, status=JobState.PENDING)

    def _run(self) -> None:
        while True:
            try:
                queued = self._queue.get(timeout=0.1)
            except Empty:
                continue

            if queued is None:
                self._queue.task_done()
                return

            self._set_job_state(queued.job_id, JobState.RUNNING)
            try:
                if queued.kind == "rewrite":
                    result = self._gateway.rewrite_listing(queued.payload)
                else:
                    result = self._gateway.extract_attributes(queued.payload)
                self._set_job_state(queued.job_id, JobState.SUCCEEDED, result=result)
                if self._config.enable_cache:
                    self._cache[queued.cache_key] = result
            except Exception as error:  # pragma: no cover - defensive runtime path
                self._set_job_state(queued.job_id, JobState.FAILED, error=str(error))
            finally:
                self._queue.task_done()

    def _set_job_state(
        self,
        job_id: str,
        status: JobState,
        *,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            self._jobs[job_id] = JobStatusResponse(job_id=job_id, status=status, result=result, error=error)


def create_low_cost_app(
    gateway: InferenceGateway | None = None,
    config: LowCostServingConfig | None = None,
    gemma_service: Any | None = None,
) -> FastAPI:
    runtime = _JobRuntime(gateway or StubLowCostGateway(), config or LowCostServingConfig())

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # CUDA Environment Check at Startup
        import torch
        print("\n" + "=" * 70)
        print("🚀 GEMMA MODEL SERVING - GPU/CUDA STATUS")
        print("=" * 70)
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✅ GPU Count: {torch.cuda.device_count()}")
            print(f"✅ GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"✅ CUDA Version: {torch.version.cuda}")
            print("✅ STATUS: GPU acceleration ENABLED 🚀")
        else:
            print("❌ GPU Count: 0")
            print("❌ GPU Name: N/A")
            print("❌ CUDA Version: Not Available")
            print("⚠️  WARNING: Running on CPU ONLY - Expect VERY SLOW inference! 🐌")
            print("   💡 Install CUDA-enabled PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        
        print("=" * 70 + "\n")
        
        runtime.start()
        yield
        runtime.stop()

    app = FastAPI(title="Gemma Model Serving", lifespan=lifespan)

    _gemma_service = gemma_service
    _active_model_id: str | None = None

    def _get_gemma_service(model_id: str | None = None):
        nonlocal _gemma_service, _active_model_id
        from gemma_serving.config import ServingConfig
        from gemma_serving.gemma_service import GemmaService

        if model_id and model_id != _active_model_id:
            _gemma_service = GemmaService(ServingConfig(model_id=model_id))
            _active_model_id = model_id
        elif _gemma_service is None:
            config = ServingConfig()
            _gemma_service = GemmaService(config)
            _active_model_id = config.model_id
        return _gemma_service

    @app.get("/health")
    def health() -> dict[str, Any]:
        service = _gemma_service
        return {
            "status": "ok",
            "active_model_id": _active_model_id,
            "model_loaded": service is not None and service.is_model_loaded(),
            "queue_size": runtime.queue_size(),
            "cache_enabled": runtime._config.enable_cache,
            "gateway": runtime.gateway_name(),
        }

    @app.post("/models/load", response_model=LoadModelResponse)
    def load_model(request: LoadModelRequest) -> LoadModelResponse:
        service = _get_gemma_service(request.model_id)
        try:
            service.ensure_loaded()
            return LoadModelResponse(
                model_id=request.model_id,
                status="ready",
                message=f"Model {request.model_id} loaded and ready.",
            )
        except Exception as error:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load model {request.model_id}: {error}",
            )

    @app.post("/generate", response_model=GenerateResponse)
    def generate(request: GenerateRequest) -> GenerateResponse:
        from gemma_serving.config import GenerationSettings

        service = _get_gemma_service(request.model_id)
        settings = GenerationSettings(
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            max_new_tokens=request.max_new_tokens,
            enable_thinking=request.enable_thinking,
            stream_output=False,
        )
        result = service.generate(request.messages, settings)
        return GenerateResponse(
            text=result["text"],
            input_token_count=result.get("input_token_count"),
            output_token_count=result.get("output_token_count"),
            total_token_count=result.get("total_token_count"),
            metadata=result.get("metadata", {}),
        )

    @app.post("/jobs/rewrite", response_model=JobAcceptedResponse)
    def submit_rewrite(request: ListingRewriteRequest) -> JobAcceptedResponse:
        return runtime.enqueue_rewrite(request)

    @app.post("/jobs/extract-attributes", response_model=JobAcceptedResponse)
    def submit_attribute_extraction(request: AttributeExtractionRequest) -> JobAcceptedResponse:
        return runtime.enqueue_attribute_extraction(request)

    @app.get("/jobs/{job_id}", response_model=JobStatusResponse)
    def get_job(job_id: str) -> JobStatusResponse:
        try:
            return runtime.get_job(job_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Job not found") from error

    return app


def create_demo_app() -> FastAPI:
    from gemma_serving.gateway import build_gateway_from_env

    return create_low_cost_app(gateway=build_gateway_from_env())


app = create_demo_app()


def _build_cache_key(kind: str, payload: dict[str, Any]) -> str:
    serialized = json.dumps({"kind": kind, "payload": payload}, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
