from __future__ import annotations

from dataclasses import dataclass, field


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