from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path


DEFAULT_MODEL_ID = os.getenv("MODEL_ID", "google/gemma-4-E2B-it")
DEFAULT_SYSTEM_PROMPT = os.getenv(
    "MODEL_SYSTEM_PROMPT",
    "You are a helpful assistant. Answer with concise, robust, direct, and short responses.",
)
SERVING_URL = os.getenv("MODEL_SERVING_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# Shared media directory
# ---------------------------------------------------------------------------
# All uploaded files (images, video, audio, PDFs, etc.) are written here.
# This folder must be accessible from BOTH:
#   • Windows (this process): as a Windows path, e.g. C:\ai-workbench\shared-media
#   • WSL2 / vLLM (the model backend): as /mnt/c/ai-workbench/shared-media
# When the paths match, the model can read files directly via file:// URIs
# instead of receiving base64-encoded payloads.
#
# Set SHARED_MEDIA_DIR in ui/.env (Windows path).
# Set SHARED_MEDIA_DIR in vllm-serving/.env.vllm (the equivalent WSL path).
# They must point to the same physical folder on disk.
SHARED_MEDIA_DIR: Path = Path(
    os.getenv("SHARED_MEDIA_DIR", r"C:\ai-workbench\shared-media")
)


def shared_media_dir_wsl() -> str:
    """Return the WSL2 equivalent of SHARED_MEDIA_DIR.

    Converts ``C:\\foo\\bar`` → ``/mnt/c/foo/bar``.
    Returns the path unchanged if it is already a POSIX path.
    """
    p = str(SHARED_MEDIA_DIR)
    if len(p) >= 2 and p[1] == ":":
        drive = p[0].lower()
        rest = p[2:].replace("\\", "/").lstrip("/")
        return f"/mnt/{drive}/{rest}"
    return p  # already POSIX (Linux native)


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
