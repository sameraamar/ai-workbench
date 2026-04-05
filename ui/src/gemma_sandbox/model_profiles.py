"""Model capability profiles.

Each entry in MODEL_PROFILES maps a user-facing display label to the HuggingFace
model ID and a ModelCapabilities descriptor that gates which media inputs the UI
exposes for that model.

Adding a new model only requires adding one entry here — no other file needs
to change for UI gating or model-ID resolution to reflect the new model.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelCapabilities:
    """Which media modalities the model accepts as input."""

    image: bool = False
    audio: bool = False
    video: bool = False
    # Approximate VRAM required in GB at BF16 (full precision).
    # Use this to warn users when the model may not fit in their GPU.
    vram_gb_bf16: float = 0.0
    # When True the model requires vLLM (e.g. AWQ quantisation) and is
    # hidden from the dropdown when the backend is the native Transformers
    # server.
    vllm_only: bool = False


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Keys are the display labels shown in the model selector dropdown.
# Values are (hf_model_id, ModelCapabilities).
MODEL_PROFILES: dict[str, tuple[str, ModelCapabilities]] = {
    # Gemma 4 — MoE family (image + audio + video)
    # MoE: effective params are small (2B/4B active per token); total stored = 5.1B/8B
    "Gemma 4 E2B IT": (
        "google/gemma-4-E2B-it",
        ModelCapabilities(image=True, audio=True, video=True, vram_gb_bf16=11.0),
    ),
    "Gemma 4 E4B IT": (
        "google/gemma-4-E4B-it",
        ModelCapabilities(image=True, audio=True, video=True, vram_gb_bf16=18.0),
    ),
    "Gemma 4 26B A4B IT": (
        "google/gemma-4-26B-A4B-it",
        ModelCapabilities(image=True, audio=False, video=True, vram_gb_bf16=52.0),
    ),
    "Gemma 4 31B IT": (
        "google/gemma-4-31B-it",
        ModelCapabilities(image=True, audio=False, video=True, vram_gb_bf16=62.0),
    ),
    # Mistral Small — image only (no audio, no video)
    "Mistral Small 3.1 (24B)": (
        "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        ModelCapabilities(image=True, audio=False, video=False, vram_gb_bf16=48.0),
    ),
    "Mistral Small 3.1 AWQ (24B)": (
        "solidrust/Mistral-Small-3.1-24B-Instruct-2503-AWQ",
        ModelCapabilities(image=True, audio=False, video=False, vram_gb_bf16=12.0, vllm_only=True),
    ),
    "Mistral Small 4 (119B)": (
        "mistralai/Mistral-Small-4-119B-2603",
        ModelCapabilities(image=True, audio=False, video=False, vram_gb_bf16=238.0),
    ),
}


def get_model_id(label: str, fallback: str = "") -> str:
    """Return the HuggingFace model ID for a display label."""
    entry = MODEL_PROFILES.get(label)
    return entry[0] if entry else fallback


def get_label_for_model_id(model_id: str) -> str | None:
    """Reverse-lookup: return the display label for a HuggingFace model ID.

    Returns ``None`` if *model_id* doesn't match any profile.
    """
    for label, (hf_id, _caps) in MODEL_PROFILES.items():
        if hf_id == model_id:
            return label
    return None


def get_capabilities(label: str) -> ModelCapabilities:
    """Return the capability descriptor for a display label."""
    entry = MODEL_PROFILES.get(label)
    return entry[1] if entry else ModelCapabilities(image=True, audio=True, video=True)


# Labels excluded from the UI selector until the backend integration is ready.
# The entries remain in MODEL_PROFILES so capability lookups and tests still work.
DISABLED_LABELS: frozenset[str] = frozenset({
    # Mistral Small 4 (119B) requires multi-GPU — disabled until verified.
    "Mistral Small 4 (119B)",
})

# Ordered list of display labels for the dropdown (excludes disabled models).
MODEL_LABELS: list[str] = [k for k in MODEL_PROFILES if k not in DISABLED_LABELS]


def model_labels_for_backend(backend_mode: str) -> list[str]:
    """Return the dropdown labels appropriate for the active backend.

    When the backend is ``"native"`` (Windows Transformers), models tagged
    ``vllm_only=True`` are hidden because they require vLLM-specific
    quantisation support (e.g. AWQ).
    """
    if backend_mode == "native":
        return [k for k in MODEL_LABELS if not MODEL_PROFILES[k][1].vllm_only]
    return list(MODEL_LABELS)
