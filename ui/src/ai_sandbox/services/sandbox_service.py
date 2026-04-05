from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ai_sandbox.config import AppConfig
from ai_sandbox.domain import RunResult

from .serving_client import ServingClient

ProgressCallback = Callable[[str, float, str], None]
TokenCallback = Callable[[str], None]


@dataclass
class TurnAttachment:
    """Media attached to a single user turn.

    Any combination of images, one audio file, and video frames is valid.
    Media parts are appended to the content list before the text part so
    the model processor sees them in a predictable order.

    URL-based media (``image_urls``, ``audio_url``, ``video_url``) is mixed
    in alongside path-based media. HuggingFace processors accept https://
    URLs directly for images; audio/video URLs are expected to already be
    resolved to a local path by the caller before setting ``audio_path`` or
    ``video_frame_paths``.
    """

    image_paths: list[Path] = field(default_factory=list)
    audio_path: Path | None = None
    video_frame_paths: list[Path] = field(default_factory=list)
    # URL-based inputs — image URLs are passed straight to the processor;
    # audio/video URL download to tmp is handled by the UI layer.
    image_urls: list[str] = field(default_factory=list)

    @property
    def has_media(self) -> bool:
        return bool(
            self.image_paths
            or self.audio_path
            or self.video_frame_paths
            or self.image_urls
        )


class SandboxService:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._client = ServingClient(base_url=config.serving_url)
        self._last_health_ok: bool | None = None

    def is_model_loaded(self) -> bool:
        healthy = self._client.is_healthy()
        self._last_health_ok = healthy
        return healthy

    def is_model_ready(self) -> bool:
        return self._client.is_model_ready()

    def get_active_model_id(self) -> str | None:
        return self._client.get_active_model_id()

    def run(
        self,
        user_prompt: str,
        attachment: TurnAttachment,
        prior_turns: list[dict],
        progress_callback: ProgressCallback | None = None,
        token_callback: TokenCallback | None = None,
    ) -> RunResult:
        was_cold_start = not (self._last_health_ok or False)

        # Build the current turn's content: media parts first, then the text prompt.
        current_content: list[dict] = []
        for path in attachment.image_paths:
            current_content.append({"type": "image", "url": path.as_posix()})
        for url in attachment.image_urls:
            current_content.append({"type": "image", "url": url})
        if attachment.audio_path is not None:
            current_content.append({"type": "audio", "audio": attachment.audio_path.as_posix()})
        for path in attachment.video_frame_paths:
            current_content.append({"type": "image", "url": path.as_posix()})
        current_content.append({"type": "text", "text": user_prompt})

        # Assemble the full message list: system + full prior history + current turn.
        messages: list[dict] = [
            {"role": "system", "content": [{"type": "text", "text": self._config.system_prompt}]},
        ]
        messages.extend(prior_turns)
        messages.append({"role": "user", "content": current_content})

        settings = self._config.generation
        response = self._client.generate(
            messages,
            model_id=self._config.model_id,
            max_new_tokens=settings.max_new_tokens,
            temperature=settings.temperature,
            top_p=settings.top_p,
            top_k=settings.top_k,
            enable_thinking=settings.enable_thinking,
            progress_callback=progress_callback,
            token_callback=token_callback,
        )
        return RunResult(
            title="Sandbox Run",
            support_level="Native",
            response_text=response["text"],
            prompt_used=user_prompt,
            model_id=self._config.model_id,
            was_cold_start=was_cold_start,
            input_token_count=response.get("input_token_count"),
            output_token_count=response.get("output_token_count"),
            total_token_count=response.get("total_token_count"),
            run_metadata=response.get("metadata", {}),
        )
