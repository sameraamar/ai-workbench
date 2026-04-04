from __future__ import annotations

import logging
from typing import Any, Callable

import httpx

LOGGER = logging.getLogger(__name__)

TokenCallback = Callable[[str], None]
ProgressCallback = Callable[[str, float, str], None]

DEFAULT_SERVING_URL = "http://localhost:8000"
GENERATE_TIMEOUT_SECONDS = 300.0
MODEL_LOAD_TIMEOUT_SECONDS = 600.0


class ServingClient:
    def __init__(self, base_url: str = DEFAULT_SERVING_URL) -> None:
        self._base_url = base_url.rstrip("/")

    def health(self) -> dict[str, Any]:
        with httpx.Client(base_url=self._base_url) as client:
            response = client.get("/health")
            response.raise_for_status()
            return response.json()

    def is_healthy(self) -> bool:
        try:
            health = self.health()
            return health.get("status") == "ok"
        except Exception:
            return False

    def get_active_model_id(self) -> str | None:
        try:
            health = self.health()
            return health.get("active_model_id")
        except Exception:
            return None

    def is_model_ready(self) -> bool:
        try:
            health = self.health()
            return health.get("model_loaded", False) is True
        except Exception:
            return False

    def load_model(self, model_id: str) -> dict[str, Any]:
        with httpx.Client(base_url=self._base_url, timeout=MODEL_LOAD_TIMEOUT_SECONDS) as client:
            response = client.post("/models/load", json={"model_id": model_id})
            response.raise_for_status()
            return response.json()

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        model_id: str | None = None,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 64,
        enable_thinking: bool = False,
        progress_callback: ProgressCallback | None = None,
        token_callback: TokenCallback | None = None,
    ) -> dict[str, Any]:
        if progress_callback:
            progress_callback("request", 0.10, "Sending request to model server...")

        payload = {
            "messages": messages,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "enable_thinking": enable_thinking,
            "stream_output": False,
        }
        if model_id is not None:
            payload["model_id"] = model_id

        with httpx.Client(base_url=self._base_url, timeout=GENERATE_TIMEOUT_SECONDS) as client:
            response = client.post("/generate", json=payload)
            response.raise_for_status()

        data = response.json()

        if progress_callback:
            progress_callback("complete", 1.0, "Model server response received.")

        return {
            "text": data.get("text", ""),
            "input_token_count": data.get("input_token_count"),
            "output_token_count": data.get("output_token_count"),
            "total_token_count": data.get("total_token_count"),
            "metadata": data.get("metadata", {}),
        }
