from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Ability(str, Enum):
    TEXT_TO_TEXT = "text-to-text"
    IMAGE_TO_TEXT = "image-to-text"
    TEXT_TO_IMAGE = "text-to-image"
    VIDEO_TO_TEXT = "video-to-text"
    TEXT_TO_VIDEO = "text-to-video"
    AUDIO_TO_TEXT = "audio-to-text"
    TEXT_TO_AUDIO = "text-to-audio"


@dataclass(frozen=True)
class AbilitySpec:
    ability: Ability
    label: str
    support_level: str
    summary: str


@dataclass
class RunResult:
    title: str
    support_level: str
    response_text: str
    prompt_used: str
    model_id: str
    was_cold_start: bool
    input_token_count: int | None = None
    output_token_count: int | None = None
    total_token_count: int | None = None
    run_metadata: dict[str, object] = field(default_factory=dict)