from __future__ import annotations

from dataclasses import dataclass, field
import os


DEFAULT_MODEL_ID = os.getenv("GEMMA_MODEL_ID", "google/gemma-4-E2B-it")
DEFAULT_SYSTEM_PROMPT = os.getenv(
    "GEMMA_SYSTEM_PROMPT",
    "You are a helpful assistant. Answer with concise, robust, direct, and short responses. Do not elaborate unless the user asks for more detail.",
)
SERVING_URL = os.getenv("GEMMA_SERVING_URL", "http://localhost:8000")


@dataclass
class GenerationSettings:
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64
    max_new_tokens: int = 256
    enable_thinking: bool = False
    stream_output: bool = True


@dataclass
class AppConfig:
    model_id: str = field(default_factory=lambda: DEFAULT_MODEL_ID)
    system_prompt: str = field(default_factory=lambda: DEFAULT_SYSTEM_PROMPT)
    simulator_name: str = "Multimodal Situation Room"
    generation: GenerationSettings = field(default_factory=GenerationSettings)
    serving_url: str = field(default_factory=lambda: SERVING_URL)
