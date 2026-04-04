from __future__ import annotations

from dataclasses import dataclass, field
import os


DEFAULT_MODEL_ID = os.getenv("GEMMA_MODEL_ID", "google/gemma-4-E2B-it")
MODEL_CACHE_DIR: str | None = os.getenv("GEMMA_MODEL_CACHE_DIR", None)
FORCE_DOWNLOAD: bool = os.getenv("GEMMA_FORCE_DOWNLOAD", "0").strip().lower() in ("1", "true", "yes")
QUANTIZE_4BIT: bool = os.getenv("GEMMA_QUANTIZE_4BIT", "0").strip().lower() in ("1", "true", "yes")

# GPU Configuration
FORCE_CPU: bool = os.getenv("GEMMA_FORCE_CPU", "0").strip().lower() in ("1", "true", "yes")
GPU_ID: int | None = int(os.getenv("GEMMA_GPU_ID", "-1")) if os.getenv("GEMMA_GPU_ID", "-1") != "-1" else None
DEVICE_MAP: str = os.getenv("GEMMA_DEVICE_MAP", "auto")  # "auto", "cpu", or specific mapping

# Performance Optimization Configuration
ENABLE_TORCH_COMPILE: bool = os.getenv("GEMMA_TORCH_COMPILE", "1").strip().lower() in ("1", "true", "yes")
ENABLE_FLASH_ATTENTION: bool = os.getenv("GEMMA_FLASH_ATTENTION", "1").strip().lower() in ("1", "true", "yes")
ENABLE_MEMORY_OPTIMIZATIONS: bool = os.getenv("GEMMA_MEMORY_OPT", "1").strip().lower() in ("1", "true", "yes")
OPTIMIZE_FOR_INFERENCE: bool = os.getenv("GEMMA_INFERENCE_OPT", "1").strip().lower() in ("1", "true", "yes")
TORCH_COMPILE_MODE: str = os.getenv("GEMMA_COMPILE_MODE", "default")  # "default", "reduce-overhead", "max-autotune"


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
