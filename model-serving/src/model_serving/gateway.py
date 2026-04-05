from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any

from model_serving.config import ServingConfig

# Lazy import to avoid triggering heavy transformers imports at app startup.
# ModelService is only needed when the real gateway is used (not the stub).
if TYPE_CHECKING:
    from model_serving.model_service import ModelService

if TYPE_CHECKING:
    from model_serving.app import AttributeExtractionRequest, ListingRewriteRequest

LOGGER = logging.getLogger(__name__)


class ModelGateway:
    """Model-agnostic gateway for marketplace operations."""

    def __init__(
        self,
        config: ServingConfig | None = None,
        model_service: ModelService | None = None,
        # Backward-compat kwarg
        gemma_service: ModelService | None = None,
    ) -> None:
        self._config = config or ServingConfig()
        svc = model_service or gemma_service
        if svc is not None:
            self._model = svc
        else:
            from model_serving.model_service import ModelService as _MS
            self._model = _MS(self._config)

    def rewrite_listing(self, request: ListingRewriteRequest) -> dict[str, Any]:
        prompt = _build_rewrite_prompt(request)
        response = self._model.generate(
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": _rewrite_system_prompt()}],
                },
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ],
            self._config.generation,
        )
        parsed = _parse_json_object(response["text"])
        return {
            "title": str(parsed.get("title", request.title)).strip()[:80],
            "description": str(parsed.get("description", request.description)).strip(),
            "marketplace": request.marketplace,
            "category_hint": request.category_hint,
            "model_id": self._config.model_id,
            "input_token_count": response.get("input_token_count"),
            "output_token_count": response.get("output_token_count"),
            "total_token_count": response.get("total_token_count"),
            "raw_text": response["text"],
        }

    def extract_attributes(self, request: AttributeExtractionRequest) -> dict[str, Any]:
        inspected_images = request.image_urls[: request.max_images]
        prompt = _build_attribute_prompt(request, inspected_images)
        response = self._model.generate(
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": _attribute_system_prompt()}],
                },
                {
                    "role": "user",
                    "content": [
                        *({"type": "image", "url": image_url} for image_url in inspected_images),
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            self._config.generation,
        )
        parsed = _parse_json_object(response["text"])
        return {
            "inspected_images": inspected_images,
            "attribute_hints": request.attribute_hints,
            "suggested_attributes": parsed.get("suggested_attributes", parsed),
            "model_id": self._config.model_id,
            "input_token_count": response.get("input_token_count"),
            "output_token_count": response.get("output_token_count"),
            "total_token_count": response.get("total_token_count"),
            "raw_text": response["text"],
        }


# Backward-compat alias
GemmaLowCostGateway = ModelGateway


def build_gateway_from_env() -> ModelGateway | None:
    gateway_mode = (
        os.getenv("MODEL_GATEWAY")
        or os.getenv("GEMMA_FASTAPI_GATEWAY", "stub")
    ).strip().lower()
    if gateway_mode in ("gemma", "model", "real"):
        return ModelGateway()
    return None


def _rewrite_system_prompt() -> str:
    return (
        "You rewrite marketplace listings accurately. "
        "Return valid JSON only with keys title and description. "
        "Keep the title concise, factual, and suitable for eBay. "
        "Do not invent product facts."
    )


def _attribute_system_prompt() -> str:
    return (
        "You extract visible product attributes from product images. "
        "Return valid JSON only with a key suggested_attributes whose value is an object. "
        "Only include attributes that are visible or strongly supported by the provided evidence."
    )


def _build_rewrite_prompt(request: ListingRewriteRequest) -> str:
    lines = [
        f"Marketplace: {request.marketplace}",
        f"Category hint: {request.category_hint or 'unspecified'}",
        "Rewrite the following product listing for marketplace publishing.",
        f"Original title: {request.title}",
        f"Original description: {request.description}",
        "Title requirements: under 80 characters, keyword-rich, factual, no hype.",
        "Description requirements: clearer wording, keep material facts, no unsupported claims.",
    ]
    return "\n".join(lines)


def _build_attribute_prompt(request: AttributeExtractionRequest, inspected_images: list[str]) -> str:
    hints = ", ".join(request.attribute_hints) if request.attribute_hints else "none"
    lines = [
        "Inspect the provided product images and extract visible product attributes.",
        f"Attribute hints: {hints}",
        f"Image count provided for analysis: {len(inspected_images)}",
        "Return only JSON.",
    ]
    return "\n".join(lines)


def _parse_json_object(raw_text: str) -> dict[str, Any]:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", maxsplit=1)[1]
        cleaned = cleaned.rsplit("```", maxsplit=1)[0].strip()

    candidate = cleaned
    if "{" in cleaned and "}" in cleaned:
        candidate = cleaned[cleaned.find("{") : cleaned.rfind("}") + 1]

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
