from __future__ import annotations

from gemma_sandbox.config import AppConfig
from gemma_sandbox.domain import Ability
from gemma_sandbox.services.sandbox_service import SandboxService


def test_text_run_includes_prior_conversation_messages(monkeypatch) -> None:
    service = SandboxService(AppConfig())
    captured: dict[str, object] = {}

    def fake_generate(messages, *, model_id=None, max_new_tokens=256, temperature=1.0, top_p=0.95, top_k=64, enable_thinking=False, progress_callback=None, token_callback=None):
        captured["messages"] = messages
        captured["model_id"] = model_id
        return {
            "text": "Follow-up answer",
            "input_token_count": 10,
            "output_token_count": 4,
            "total_token_count": 14,
            "metadata": {},
        }

    monkeypatch.setattr(service._client, "generate", fake_generate)
    monkeypatch.setattr(service, "_last_health_ok", True)

    prior_messages = [
        {"role": "user", "content": [{"type": "text", "text": "First question with preset framing."}]},
        {"role": "assistant", "content": [{"type": "text", "text": "First answer."}]},
    ]

    result = service.run(
        ability=Ability.TEXT_TO_TEXT,
        user_prompt="Second question",
        prior_messages=prior_messages,
    )

    messages = captured["messages"]
    assert isinstance(messages, list)
    assert messages[0]["role"] == "system"
    assert messages[1:3] == prior_messages
    assert messages[3]["role"] == "user"
    assert "Second question" in messages[3]["content"][0]["text"]
    assert result.response_text == "Follow-up answer"