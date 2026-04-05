from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI

LOGGER = logging.getLogger(__name__)

TokenCallback = Callable[[str], None]
ProgressCallback = Callable[[str, float, str], None]

DEFAULT_SERVING_URL = "http://localhost:8000"
GENERATE_TIMEOUT_SECONDS = 300.0


class ServingClient:
    """Talks to a vLLM (or any OpenAI-compatible) server via the ``openai`` SDK.

    Endpoints used:
      GET  /health                — liveness check (raw HTTP, not part of OpenAI spec)
      GET  /v1/models             — list loaded models
      POST /v1/chat/completions   — chat generation (+ SSE streaming)
    """

    def __init__(self, base_url: str = DEFAULT_SERVING_URL) -> None:
        self._base_url = base_url.rstrip("/")
        # The OpenAI client adds /v1 automatically; point it at the bare base URL.
        # api_key is required by the SDK but irrelevant for local vLLM.
        self._client = OpenAI(
            base_url=f"{self._base_url}/v1",
            api_key="not-needed",
            timeout=GENERATE_TIMEOUT_SECONDS,
        )

    # ------------------------------------------------------------------
    # Health / readiness
    # ------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        """Return a health dict.  vLLM's ``/health`` returns 200 with empty body."""
        import httpx

        with httpx.Client(base_url=self._base_url, timeout=5.0) as http:
            response = http.get("/health")
            response.raise_for_status()
        return {"status": "ok"}

    def is_healthy(self) -> bool:
        try:
            self.health()
            return True
        except Exception:
            return False

    def get_active_model_id(self) -> str | None:
        """Return the first model ID advertised by the server."""
        try:
            page = self._client.models.list()
            models = list(page)
            if models:
                return models[0].id
            return None
        except Exception:
            return None

    def is_model_ready(self) -> bool:
        """vLLM always has its model ready once the server is up."""
        return self.is_healthy()

    def detect_backend_mode(self) -> str:
        """Return ``"native"`` for the FastAPI/Transformers backend or ``"vllm"`` for vLLM.

        Heuristic: the native backend's ``/health`` returns JSON with a ``gateway``
        key, while vLLM's ``/health`` returns HTTP 200 with an empty body.
        """
        import httpx

        try:
            with httpx.Client(base_url=self._base_url, timeout=5.0) as http:
                resp = http.get("/health")
                resp.raise_for_status()
                body = resp.text.strip()
                if body:
                    data = resp.json()
                    if "gateway" in data:
                        return "native"
                return "vllm"
        except Exception:
            return "vllm"  # default assumption when unreachable

    def load_model(self, model_id: str) -> dict[str, Any]:
        """Ask the Windows backend to load a specific model via ``POST /models/load``."""
        import httpx

        with httpx.Client(base_url=self._base_url, timeout=300.0) as http:
            resp = http.post("/models/load", json={"model_id": model_id})
            resp.raise_for_status()
            return resp.json()

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

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
            progress_callback("request", 0.10, "Sending request to server...")

        # Resolve model ID — vLLM requires ``model`` in every request.
        resolved_model = model_id or self.get_active_model_id() or "default"

        openai_messages = _to_openai_messages(messages)

        # Build optional extra_body for vLLM-specific params.
        extra_body: dict[str, Any] = {}
        if top_k != 64:
            extra_body["top_k"] = top_k
        if enable_thinking:
            extra_body["enable_thinking"] = True

        stream = token_callback is not None

        if stream:
            return self._generate_streaming(
                openai_messages,
                resolved_model,
                max_new_tokens,
                temperature,
                top_p,
                extra_body or None,
                progress_callback,
                token_callback,
            )
        else:
            return self._generate_one_shot(
                openai_messages,
                resolved_model,
                max_new_tokens,
                temperature,
                top_p,
                extra_body or None,
                progress_callback,
            )

    # ------------------------------------------------------------------
    # One-shot (non-streaming)
    # ------------------------------------------------------------------

    def _generate_one_shot(
        self,
        messages: list[dict[str, Any]],
        model: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        extra_body: dict[str, Any] | None,
        progress_callback: ProgressCallback | None,
    ) -> dict[str, Any]:
        completion = self._client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            extra_body=extra_body,
            stream=False,
        )

        text = ""
        if completion.choices:
            text = completion.choices[0].message.content or ""

        usage = completion.usage
        if progress_callback:
            progress_callback("complete", 1.0, "Response received.")

        return {
            "text": text,
            "input_token_count": usage.prompt_tokens if usage else None,
            "output_token_count": usage.completion_tokens if usage else None,
            "total_token_count": usage.total_tokens if usage else None,
            "metadata": {"model_id": completion.model or model},
        }

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def _generate_streaming(
        self,
        messages: list[dict[str, Any]],
        model: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        extra_body: dict[str, Any] | None,
        progress_callback: ProgressCallback | None,
        token_callback: TokenCallback,
    ) -> dict[str, Any]:
        stream = self._client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            extra_body=extra_body,
            stream=True,
            stream_options={"include_usage": True},
        )

        collected_text = ""
        model_id = model
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        total_tokens: int | None = None

        if progress_callback:
            progress_callback("stream", 0.40, "Receiving generated tokens...")

        for chunk in stream:
            if not model_id and chunk.model:
                model_id = chunk.model

            # Usage is included in the final chunk when stream_options.include_usage is set.
            if chunk.usage:
                prompt_tokens = chunk.usage.prompt_tokens
                completion_tokens = chunk.usage.completion_tokens
                total_tokens = chunk.usage.total_tokens

            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    collected_text += delta.content
                    token_callback(collected_text)

        if progress_callback:
            progress_callback("complete", 1.0, "Response received.")

        return {
            "text": collected_text,
            "input_token_count": prompt_tokens,
            "output_token_count": completion_tokens,
            "total_token_count": total_tokens,
            "metadata": {"model_id": model_id},
        }


# ---------------------------------------------------------------------------
# Message format conversion
# ---------------------------------------------------------------------------

def _to_openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert our internal message format to OpenAI chat API format.

    Input format (our internal):
      {"role": "user", "content": [
          {"type": "text", "text": "..."},
          {"type": "image", "url": "..."},
          {"type": "audio", "audio": "..."},
      ]}

    Output format (OpenAI / vLLM):
      {"role": "user", "content": [
          {"type": "text", "text": "..."},
          {"type": "image_url", "image_url": {"url": "..."}},
      ]}

    For plain string content, pass through as-is.
    """
    converted: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            converted.append({"role": msg["role"], "content": content})
            continue
        if isinstance(content, list):
            parts: list[dict[str, Any]] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type", "")
                if block_type == "text":
                    parts.append({"type": "text", "text": block.get("text", "")})
                elif block_type == "image":
                    url = block.get("url", "")
                    parts.append({
                        "type": "image_url",
                        "image_url": {"url": _ensure_data_uri_or_url(url)},
                    })
                elif block_type == "audio":
                    # vLLM does not have a standard audio content type in
                    # the OpenAI schema; pass the path as text context for now.
                    audio_path = block.get("audio", "")
                    parts.append({
                        "type": "text",
                        "text": f"[Audio file attached: {audio_path}]",
                    })
                else:
                    # Unknown block type — pass through
                    parts.append(block)
            converted.append({"role": msg["role"], "content": parts})
        else:
            # Fallback
            converted.append(msg)
    return converted


def _ensure_data_uri_or_url(url: str) -> str:
    """If url is a local file path, convert to a data URI. Otherwise pass through."""
    # Already a URL — pass through
    if url.startswith("http://") or url.startswith("https://") or url.startswith("data:"):
        return url

    # Try as local file path
    path = Path(url)
    if path.is_file():
        suffix = path.suffix.lower()
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }
        mime = mime_map.get(suffix, "image/png")
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{data}"

    # Can't resolve — return as-is (server will error if invalid)
    return url
