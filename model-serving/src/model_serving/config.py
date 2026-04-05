from __future__ import annotations

from dataclasses import dataclass, field
import os


def _env(new: str, old: str, default: str = "") -> str:
    """Read *new* env-var, fall back to *old* for backward compat."""
    return os.getenv(new) or os.getenv(old, default)


def _env_bool(new: str, old: str, default: str = "0") -> bool:
    return _env(new, old, default).strip().lower() in ("1", "true", "yes")


DEFAULT_MODEL_ID = _env("MODEL_ID", "GEMMA_MODEL_ID", "google/gemma-4-E2B-it")
MODEL_CACHE_DIR: str | None = _env("MODEL_CACHE_DIR", "GEMMA_MODEL_CACHE_DIR") or None
FORCE_DOWNLOAD: bool = _env_bool("MODEL_FORCE_DOWNLOAD", "GEMMA_FORCE_DOWNLOAD")
QUANTIZE_4BIT: bool = _env_bool("MODEL_QUANTIZE_4BIT", "GEMMA_QUANTIZE_4BIT")

# GPU Configuration
FORCE_CPU: bool = _env_bool("MODEL_FORCE_CPU", "GEMMA_FORCE_CPU")
GPU_ID: int | None = None
gpu_id_str = _env("MODEL_GPU_ID", "GEMMA_GPU_ID").strip()
if gpu_id_str and gpu_id_str != "-1":
    try:
        GPU_ID = int(gpu_id_str)
    except ValueError:
        GPU_ID = None
DEVICE_MAP: str = _env("MODEL_DEVICE_MAP", "GEMMA_DEVICE_MAP", "auto")

# Performance Optimization Configuration
ENABLE_TORCH_COMPILE: bool = _env_bool("MODEL_TORCH_COMPILE", "GEMMA_TORCH_COMPILE", "1")
ENABLE_FLASH_ATTENTION: bool = _env_bool("MODEL_FLASH_ATTENTION", "GEMMA_FLASH_ATTENTION", "1")
ENABLE_MEMORY_OPTIMIZATIONS: bool = _env_bool("MODEL_MEMORY_OPT", "GEMMA_MEMORY_OPT", "1")
OPTIMIZE_FOR_INFERENCE: bool = _env_bool("MODEL_INFERENCE_OPT", "GEMMA_INFERENCE_OPT", "1")
TORCH_COMPILE_MODE: str = _env("MODEL_COMPILE_MODE", "GEMMA_COMPILE_MODE", "default")
# Safety cap on input tokens before generation to prevent OOM on long prompts.
MAX_INPUT_TOKENS: int = int(_env("MODEL_MAX_INPUT_TOKENS", "GEMMA_MAX_INPUT_TOKENS", "8192"))


@dataclass
class GenerationSettings:
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64
    max_new_tokens: int = 256
    enable_thinking: bool = False
    stream_output: bool = True


@dataclass
class ServingConfig:
    model_id: str = field(default_factory=lambda: DEFAULT_MODEL_ID)
    generation: GenerationSettings = field(default_factory=GenerationSettings)
    model_cache_dir: str | None = field(default_factory=lambda: MODEL_CACHE_DIR)
    force_download: bool = field(default_factory=lambda: FORCE_DOWNLOAD)
    quantize_4bit: bool = field(default_factory=lambda: QUANTIZE_4BIT)
    
    # GPU Configuration
    force_cpu: bool = field(default_factory=lambda: FORCE_CPU)
    gpu_id: int | None = field(default_factory=lambda: GPU_ID)
    device_map: str = field(default_factory=lambda: DEVICE_MAP)
    
    # Performance Optimization Configuration
    enable_torch_compile: bool = field(default_factory=lambda: ENABLE_TORCH_COMPILE)
    enable_flash_attention: bool = field(default_factory=lambda: ENABLE_FLASH_ATTENTION)
    enable_memory_optimizations: bool = field(default_factory=lambda: ENABLE_MEMORY_OPTIMIZATIONS)
    optimize_for_inference: bool = field(default_factory=lambda: OPTIMIZE_FOR_INFERENCE)
    torch_compile_mode: str = field(default_factory=lambda: TORCH_COMPILE_MODE)
    max_input_tokens: int = field(default_factory=lambda: MAX_INPUT_TOKENS)
