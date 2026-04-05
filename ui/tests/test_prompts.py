"""Tests for model_profiles.py (replaces legacy persona-preset tests)."""
from gemma_sandbox.model_profiles import (
    MODEL_LABELS,
    MODEL_PROFILES,
    ModelCapabilities,
    get_capabilities,
    get_model_id,
)


def test_model_profiles_not_empty() -> None:
    assert len(MODEL_PROFILES) >= 4


def test_all_gemma_models_support_images() -> None:
    for label, (_mid, caps) in MODEL_PROFILES.items():
        assert caps.image, f"{label} should support images"


def test_gemma_e2b_supports_audio() -> None:
    caps = get_capabilities("Gemma 4 E2B IT")
    assert caps.audio


def test_mistral_does_not_support_audio() -> None:
    caps = get_capabilities("Mistral Small 3.1 (24B)")
    assert not caps.audio


def test_mistral_does_not_support_video() -> None:
    caps = get_capabilities("Mistral Small 4 (119B)")
    assert not caps.video


def test_get_model_id_returns_hf_id() -> None:
    mid = get_model_id("Gemma 4 E2B IT")
    assert mid == "google/gemma-4-E2B-it"


def test_get_model_id_fallback_for_unknown() -> None:
    mid = get_model_id("Unknown Label", fallback="some/default")
    assert mid == "some/default"


def test_model_labels_has_all_profiles() -> None:
    assert MODEL_LABELS == list(MODEL_PROFILES.keys())
