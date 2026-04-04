from __future__ import annotations

from pathlib import Path
from typing import Callable

from gemma_sandbox.config import AppConfig
from gemma_sandbox.domain import Ability, AbilitySpec, RunResult
from gemma_sandbox.prompts import ABILITY_SPECS, build_audio_prompt, build_image_prompt, build_simulation_prompt, build_text_prompt, build_video_prompt

from .serving_client import ServingClient

ProgressCallback = Callable[[str, float, str], None]
TokenCallback = Callable[[str], None]
ConversationMessages = list[dict]


class SandboxService:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._client = ServingClient(base_url=config.serving_url)
        self._last_health_ok: bool | None = None

    def get_ability_spec(self, ability: Ability) -> AbilitySpec:
        support_level, summary = ABILITY_SPECS[ability]
        return AbilitySpec(
            ability=ability,
            label=ability.value,
            support_level=support_level,
            summary=summary,
        )

    def is_model_loaded(self) -> bool:
        healthy = self._client.is_healthy()
        self._last_health_ok = healthy
        return healthy

    def is_model_ready(self) -> bool:
        return self._client.is_model_ready()

    def get_active_model_id(self) -> str | None:
        return self._client.get_active_model_id()

    def load_model(self, model_id: str) -> dict:
        return self._client.load_model(model_id)

    def run(
        self,
        ability: Ability,
        user_prompt: str,
        uploaded_path: Path | None = None,
        frame_paths: list[Path] | None = None,
        prior_messages: ConversationMessages | None = None,
        progress_callback: ProgressCallback | None = None,
        token_callback: TokenCallback | None = None,
    ) -> RunResult:
        spec = self.get_ability_spec(ability)
        was_cold_start = not (self._last_health_ok or False)
        if ability is Ability.TEXT_TO_TEXT:
            prompt = build_text_prompt(user_prompt)
            messages = _conversation_messages(self._config.system_prompt, prior_messages, prompt)
        elif ability is Ability.IMAGE_TO_TEXT:
            if uploaded_path is None:
                raise ValueError("Image-to-text requires an image upload.")
            prompt = build_image_prompt(user_prompt)
            messages = _image_messages(self._config.system_prompt, prompt, uploaded_path)
        elif ability is Ability.AUDIO_TO_TEXT:
            if uploaded_path is None:
                raise ValueError("Audio-to-text requires an audio upload.")
            prompt = build_audio_prompt(user_prompt)
            messages = _audio_messages(self._config.system_prompt, prompt, uploaded_path)
        elif ability is Ability.VIDEO_TO_TEXT:
            if not frame_paths:
                raise ValueError("Video-to-text requires sampled frames.")
            prompt = build_video_prompt(user_prompt)
            messages = _video_messages(self._config.system_prompt, prompt, frame_paths)
        else:
            prompt = build_simulation_prompt(ability, user_prompt)
            messages = _conversation_messages(self._config.system_prompt, prior_messages, prompt)

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
            title=spec.label,
            support_level=spec.support_level,
            response_text=response["text"],
            prompt_used=prompt,
            model_id=self._config.model_id,
            was_cold_start=was_cold_start,
            input_token_count=response.get("input_token_count"),
            output_token_count=response.get("output_token_count"),
            total_token_count=response.get("total_token_count"),
            run_metadata=response.get("metadata", {}),
        )


def _text_messages(system_prompt: str, prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]


def _conversation_messages(
    system_prompt: str,
    prior_messages: ConversationMessages | None,
    prompt: str,
) -> list[dict]:
    messages = [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}]
    if prior_messages:
        messages.extend(prior_messages)
    messages.append({"role": "user", "content": [{"type": "text", "text": prompt}]})
    return messages


def _image_messages(system_prompt: str, prompt: str, image_path: Path) -> list[dict]:
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image_path.as_posix()},
                {"type": "text", "text": prompt},
            ],
        },
    ]


def _audio_messages(system_prompt: str, prompt: str, audio_path: Path) -> list[dict]:
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path.as_posix()},
                {"type": "text", "text": prompt},
            ],
        },
    ]


def _video_messages(system_prompt: str, prompt: str, frame_paths: list[Path]) -> list[dict]:
    content = [{"type": "image", "url": frame_path.as_posix()} for frame_path in frame_paths]
    content.append({"type": "text", "text": prompt})
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": content},
    ]