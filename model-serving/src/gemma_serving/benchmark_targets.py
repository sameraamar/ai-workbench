from __future__ import annotations

from functools import lru_cache
import logging
from typing import Any

from gemma_serving.config import ServingConfig, GenerationSettings

from .benchmarking import BenchmarkScenario
from .gateway import GemmaLowCostGateway
from .app import ListingRewriteRequest


LOGGER = logging.getLogger(__name__)


def benchmark_listing_rewrite(scenario: BenchmarkScenario) -> dict[str, Any]:
    metadata = scenario.metadata
    model_id = str(metadata.get("model_id", _model_id_from_label(scenario.model_label)))
    LOGGER.info(
        "Preparing live rewrite benchmark for scenario '%s' with model %s",
        scenario.name,
        model_id,
    )
    gateway = _get_gateway(
        model_id=model_id,
        max_new_tokens=int(metadata.get("max_new_tokens", 256)),
        enable_thinking=bool(metadata.get("enable_thinking", False)),
    )
    request = ListingRewriteRequest(
        title=str(metadata["title"]),
        description=str(metadata["description"]),
        marketplace=str(metadata.get("marketplace", "ebay")),
        category_hint=_optional_string(metadata.get("category_hint")),
    )
    LOGGER.info(
        "Submitting rewrite request: marketplace=%s category=%s title_length=%s description_length=%s",
        request.marketplace,
        request.category_hint or "unspecified",
        len(request.title),
        len(request.description),
    )
    return gateway.rewrite_listing(request)


@lru_cache(maxsize=8)
def _get_gateway(
    *,
    model_id: str,
    max_new_tokens: int,
    enable_thinking: bool,
) -> GemmaLowCostGateway:
    LOGGER.info(
        "Creating Gemma benchmark gateway for %s (max_new_tokens=%s, thinking=%s)",
        model_id,
        max_new_tokens,
        enable_thinking,
    )
    config = ServingConfig(
        model_id=model_id,
        generation=GenerationSettings(
            max_new_tokens=max_new_tokens,
            enable_thinking=enable_thinking,
        ),
    )
    return GemmaLowCostGateway(config=config)


def _model_id_from_label(model_label: str) -> str:
    normalized = model_label.strip().lower()
    if "e4b" in normalized:
        return "google/gemma-4-E4B-it"
    if "26b" in normalized:
        return "google/gemma-4-26B-A4B-it"
    if "31b" in normalized:
        return "google/gemma-4-31B-it"
    return "google/gemma-4-E2B-it"


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None