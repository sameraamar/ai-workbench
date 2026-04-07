"""OpenAI-compatible API shim for the Windows-native Transformers backend.

Adds ``/v1/chat/completions`` and ``/v1/models`` routes to an existing FastAPI
app so that the UI's ``serving_client.py`` (which speaks the OpenAI protocol)
can talk to *either*:

  - **vLLM** in WSL2 (native OpenAI endpoints)
  - **this shim** on Windows (wraps ModelService)

No changes needed in the UI — it hits the same endpoints on ``localhost:8000``.
"""

from __future__ import annotations

import json
import time
import logging
from typing import Any, Callable
from threading import Thread

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models (subset of the OpenAI chat completions schema)
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: Any  # str or list[dict]


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = Field(default=256, ge=1)
    temperature: float = Field(default=1.0, ge=0.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    top_k: int | None = None  # non-standard, but vLLM supports it
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    extra_body: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Message format conversion: OpenAI → internal
# ---------------------------------------------------------------------------

def _openai_to_internal_messages(
    messages: list[ChatMessage],
) -> list[dict[str, Any]]:
    """Convert OpenAI-format messages to the internal format ModelService expects.

    OpenAI format              →  Internal format
    ────────────────────────      ──────────────────────
    {"type": "image_url",      →  {"type": "image",
     "image_url": {"url": …}}      "url": …}
    """
    converted: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.content
        if isinstance(content, str):
            converted.append({"role": msg.role, "content": content})
            continue
        if isinstance(content, list):
            parts: list[dict[str, Any]] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type", "")
                if btype == "text":
                    parts.append({"type": "text", "text": block.get("text", "")})
                elif btype == "image_url":
                    url = block.get("image_url", {}).get("url", "")
                    parts.append({"type": "image", "url": url})
                elif btype == "video_path":
                    # Custom block sent by serving_client.py for native backend.
                    # Convert to the format Gemma 4's processor expects.
                    vpath = block.get("video_path", "")
                    parts.append({"type": "video", "video": vpath})
                else:
                    parts.append(block)
            converted.append({"role": msg.role, "content": parts})
        else:
            converted.append({"role": msg.role, "content": content})
    return converted


# ---------------------------------------------------------------------------
# Response builders
# ---------------------------------------------------------------------------

def _chat_completion_response(
    text: str,
    model_id: str,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
) -> dict[str, Any]:
    total = None
    if prompt_tokens is not None and completion_tokens is not None:
        total = prompt_tokens + completion_tokens
    return {
        "id": f"chatcmpl-{int(time.time()*1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens or 0,
            "completion_tokens": completion_tokens or 0,
            "total_tokens": total or 0,
        },
    }


def _sse_chunk(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _stream_delta_chunk(
    model_id: str,
    content: str,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    if content:
        delta["content"] = content
    return {
        "id": f"chatcmpl-{int(time.time()*1000)}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


def _stream_usage_chunk(
    model_id: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{int(time.time()*1000)}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_id,
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------

def register_openai_routes(
    app: FastAPI,
    get_service: Callable[[str | None], Any],
    get_active_model_id: Callable[[], str | None],
) -> None:
    """Add ``/v1/models`` and ``/v1/chat/completions`` to *app*.

    Parameters
    ----------
    app:
        The existing FastAPI application.
    get_service:
        Callable that returns a ModelService for a given model_id (or None
        for the default model).  Expected to lazy-load on first call.
    get_active_model_id:
        Returns the currently loaded model ID, or None.
    """

    # --- GET /v1/models ---------------------------------------------------

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        model_id = get_active_model_id()
        models = []
        if model_id:
            models.append(
                {
                    "id": model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "local",
                }
            )
        return {"object": "list", "data": models}

    # --- POST /v1/chat/completions ----------------------------------------

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest):
        from model_serving.config import GenerationSettings

        model_id = request.model or get_active_model_id() or "unknown"
        service = get_service(request.model)

        internal_messages = _openai_to_internal_messages(request.messages)

        top_k = request.top_k or 64
        if request.extra_body and "top_k" in request.extra_body:
            top_k = request.extra_body["top_k"]

        settings = GenerationSettings(
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=top_k,
            max_new_tokens=request.max_tokens,
            enable_thinking=False,
            stream_output=request.stream,
        )

        if not request.stream:
            # --- One-shot ------------------------------------------------
            result = service.generate(internal_messages, settings)
            return _chat_completion_response(
                text=result.get("text", ""),
                model_id=model_id,
                prompt_tokens=result.get("input_token_count"),
                completion_tokens=result.get("output_token_count"),
            )

        # --- SSE streaming -----------------------------------------------
        include_usage = False
        if request.stream_options and request.stream_options.get("include_usage"):
            include_usage = True

        def _event_stream():
            collected_text = ""
            prev_len = 0
            result: dict[str, Any] = {}

            def _token_cb(partial: str) -> None:
                nonlocal collected_text
                collected_text = partial

            result_holder: list[dict[str, Any]] = []
            error_holder: list[Exception] = []

            def _worker():
                try:
                    r = service.generate(
                        internal_messages,
                        settings,
                        token_callback=_token_cb,
                    )
                    result_holder.append(r)
                except Exception as exc:
                    LOGGER.error("SSE worker error: %s", exc, exc_info=True)
                    error_holder.append(exc)

            worker = Thread(target=_worker, daemon=True)
            worker.start()

            while worker.is_alive():
                worker.join(timeout=0.05)
                if len(collected_text) > prev_len:
                    delta = collected_text[prev_len:]
                    prev_len = len(collected_text)
                    yield _sse_chunk(
                        _stream_delta_chunk(model_id, delta)
                    )

            # Surface worker errors as an SSE error event so the client
            # doesn't silently receive an empty response.
            if error_holder:
                err_msg = str(error_holder[0])
                yield _sse_chunk(
                    _stream_delta_chunk(model_id, f"[Error: {err_msg}]")
                )
                yield _sse_chunk(
                    _stream_delta_chunk(model_id, "", finish_reason="stop")
                )
                yield "data: [DONE]\n\n"
                return

            # Flush any remaining text from streaming token callback
            if len(collected_text) > prev_len:
                delta = collected_text[prev_len:]
                prev_len = len(collected_text)
                yield _sse_chunk(
                    _stream_delta_chunk(model_id, delta)
                )

            # Multimodal one-shot fallback: _generate_multimodal does not
            # call token_callback, so the result text lives only in
            # result_holder.  Emit anything not already streamed.
            if result_holder:
                full_text = result_holder[0].get("text", "")
                if len(full_text) > prev_len:
                    yield _sse_chunk(
                        _stream_delta_chunk(model_id, full_text[prev_len:])
                    )

            # Finish chunk
            yield _sse_chunk(
                _stream_delta_chunk(model_id, "", finish_reason="stop")
            )

            # Usage chunk (if requested)
            if include_usage and result_holder:
                r = result_holder[0]
                yield _sse_chunk(
                    _stream_usage_chunk(
                        model_id,
                        prompt_tokens=r.get("input_token_count", 0) or 0,
                        completion_tokens=r.get("output_token_count", 0) or 0,
                    )
                )

            yield "data: [DONE]\n\n"

        return StreamingResponse(
            _event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
