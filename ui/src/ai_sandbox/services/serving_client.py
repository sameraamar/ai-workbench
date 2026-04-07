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
        self._client = OpenAI(
            base_url=f"{self._base_url}/v1",
            api_key="not-needed",
            timeout=GENERATE_TIMEOUT_SECONDS,
        )
        # Cached after the first generate() call — avoids an extra HTTP round-trip
        # on every request while still reflecting the actual backend.
        self._cached_backend_mode: str | None = None

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
        """Return ``"native"`` for the FastAPI/Transformers backend or ``"vllm"`` for vLLM."""
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

    def _get_backend_mode(self) -> str:
        """Return cached backend mode, detecting it on first call."""
        if self._cached_backend_mode is None:
            self._cached_backend_mode = self.detect_backend_mode()
        return self._cached_backend_mode

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

        openai_messages = _to_openai_messages(messages, self._get_backend_mode())
        # Log converted message summary for diagnostics
        for _mi, _m in enumerate(openai_messages):
            _c = _m.get("content")
            if isinstance(_c, list):
                for _bi, _b in enumerate(_c):
                    _bt = _b.get("type", "?")
                    if _bt == "image_url":
                        _url = _b.get("image_url", {}).get("url", "")
                        LOGGER.info(
                            "MSG_DIAG: msg[%d][%d] image_url len=%d starts=%r",
                            _mi, _bi, len(_url), _url[:50],
                        )

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

def _to_openai_messages(messages: list[dict[str, Any]], backend_mode: str = "vllm") -> list[dict[str, Any]]:
    """Convert our internal message format to OpenAI chat API format.

    Input format (our internal):
      {"role": "user", "content": [
          {"type": "text", "text": "..."},
          {"type": "image", "url": "..."},
          {"type": "audio", "audio": "..."},
          {"type": "video", "path": "..."},
      ]}

    Output format (OpenAI / vLLM):
      {"role": "user", "content": [
          {"type": "text", "text": "..."},
          {"type": "image_url", "image_url": {"url": "..."}},
          {"type": "video_url", "video_url": {"url": "file:///mnt/c/..."}},  # vLLM
          {"type": "video_path", "video_path": "..."},  # native backend
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
                    if backend_mode == "vllm":
                        resolved = _to_file_uri_for_vllm(url)
                    else:
                        # Native backend: send local file paths as-is so the
                        # Transformers processor can call load_image() / PIL.Image.open()
                        # directly from SHARED_MEDIA_DIR — no base64 encoding needed.
                        # Only fall back to data: URI for http(s) URLs (no local file).
                        resolved = _ensure_local_path_or_url(url)
                    if not resolved:
                        LOGGER.warning(
                            "Dropping unresolvable image from message: %s",
                            url[:80] if url else "(empty)",
                        )
                        continue
                    parts.append({
                        "type": "image_url",
                        "image_url": {"url": resolved},
                    })
                elif block_type == "audio":
                    audio_ref = block.get("audio", "")
                    if backend_mode == "vllm":
                        wsl_uri = _to_file_uri_for_vllm(audio_ref)
                        LOGGER.info("AUDIO: vLLM %s -> %s", audio_ref, wsl_uri)
                        parts.append({"type": "text", "text": f"[Audio file: {wsl_uri}]"})
                    else:
                        parts.append({"type": "text", "text": f"[Audio file attached: {audio_ref}]"})
                elif block_type == "video":
                    video_path = block.get("path", "")
                    if backend_mode == "vllm":
                        wsl_uri = _to_file_uri_for_vllm(video_path)
                        LOGGER.info("VIDEO: vLLM %s -> %s", video_path, wsl_uri)
                        parts.append({
                            "type": "video_url",
                            "video_url": {"url": wsl_uri},
                        })
                    else:
                        parts.append({
                            "type": "video_path",
                            "video_path": video_path,
                        })
                else:
                    # Unknown block type — pass through
                    parts.append(block)
            converted.append({"role": msg["role"], "content": parts})
        else:
            # Fallback
            converted.append(msg)
    return converted


def _to_file_uri_for_vllm(path: str) -> str:
    """Convert any file reference to a WSL2-accessible file:// URI.

    Works for Windows paths, POSIX paths, and pass-through for existing
    http(s):// and data: URIs.

    Examples::

        C:\\ai-workbench\\shared-media\\abc.mp4
            → file:///mnt/c/ai-workbench/shared-media/abc.mp4

        /mnt/c/tmp/foo.png
            → file:///mnt/c/tmp/foo.png

        https://example.com/img.jpg
            → https://example.com/img.jpg  (unchanged)

        data:image/png;base64,...
            → data:image/png;base64,...  (unchanged)
    """
    return _windows_path_to_wsl_file_uri(path)


def _windows_path_to_wsl_file_uri(path: str) -> str:
    """Convert a Windows absolute path to a WSL2-accessible file:// URI.

    Examples:
      C:\\temp\\foo.mp4         -> file:///mnt/c/temp/foo.mp4
      C:/Users/foo/bar.mp4    -> file:///mnt/c/Users/foo/bar.mp4
      /tmp/foo.mp4            -> file:///tmp/foo.mp4  (pass-through)
    """
    from pathlib import PurePosixPath, PureWindowsPath

    # Already a URI — pass through
    if path.startswith("file://") or path.startswith("http"):
        return path

    # Detect Windows path by drive letter (e.g. 'C:' or 'C:/')
    if len(path) >= 2 and path[1] == ":":
        drive_letter = path[0].lower()
        rest = path[2:].replace("\\", "/").lstrip("/")
        return f"file:///mnt/{drive_letter}/{rest}"

    # Already a POSIX path (e.g. running on Linux natively)
    return f"file://{path}"


def _ensure_local_path_or_url(url: str) -> str:
    """For the native (Transformers) backend: pass http/data URIs through unchanged;
    verify that local file paths exist and return them as-is so the server-side
    Transformers processor can open them via PIL.Image.open() without encoding."""
    if url.startswith("http://") or url.startswith("https://"):
        return url
    if url.startswith("data:"):
        return url
    # Local file path — verify existence, then pass directly.
    path = Path(url)
    if path.is_file():
        return url
    LOGGER.warning("Image file not found, dropping: %s", url)
    return ""


def _ensure_data_uri_or_url(url: str) -> str:
    """If url is a local file path, convert to a data URI. Otherwise pass through.
    Kept for backward compatibility; prefer _ensure_local_path_or_url() for native backend."""
    # Already a URL — pass through
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

    # Can't resolve — return empty string so _to_openai_messages can
    # drop this block.  This prevents server crashes when prior-turn temp
    # files have been cleaned up.
    LOGGER.warning("Image file not found: %s", url)
    return ""
