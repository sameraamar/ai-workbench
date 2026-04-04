from __future__ import annotations

from dataclasses import dataclass, field
import os


DEFAULT_MODEL_ID = os.getenv("GEMMA_MODEL_ID", "google/gemma-4-E2B-it")
MODEL_CACHE_DIR: str | None = os.getenv("GEMMA_MODEL_CACHE_DIR", None)
FORCE_DOWNLOAD: bool = os.getenv("GEMMA_FORCE_DOWNLOAD", "0").strip().lower() in ("1", "true", "yes")
QUANTIZE_4BIT: bool = os.getenv("GEMMA_QUANTIZE_4BIT", "0").strip().lower() in ("1", "true", "yes")


@dataclass(slots=True)
class GenerationSettings:
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64
    max_new_tokens: int = 256
    enable_thinking: bool = False
    stream_output: bool = True


@dataclass(slots=True)
class ServingConfig:
    model_id: str = field(default_factory=lambda: DEFAULT_MODEL_ID)
    generation: GenerationSettings = field(default_factory=GenerationSettings)
    model_cache_dir: str | None = field(default_factory=lambda: MODEL_CACHE_DIR)
    force_download: bool = field(default_factory=lambda: FORCE_DOWNLOAD)
    quantize_4bit: bool = field(default_factory=lambda: QUANTIZE_4BIT)
