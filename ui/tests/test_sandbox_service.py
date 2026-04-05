from __future__ import annotations

from pathlib import Path

from gemma_sandbox.config import AppConfig
from gemma_sandbox.services.sandbox_service import SandboxService, TurnAttachment


def _fake_generate(messages, *, model_id=None, max_new_tokens=256, temperature=1.0, top_p=0.95, top_k=64, enable_thinking=False, progress_callback=None, token_callback=None):
    return {
        "text": "Test answer",
        "input_token_count": 10,
        "output_token_count": 4,
        "total_token_count": 14,
        "metadata": {},
    }


def test_run_includes_prior_turns_in_messages(monkeypatch) -> None:
    service = SandboxService(AppConfig())
    captured: dict = {}

    def fake_generate(messages, **kwargs):
        captured["messages"] = messages
        return _fake_generate(messages, **kwargs)

    monkeypatch.setattr(service._client, "generate", fake_generate)
    monkeypatch.setattr(service, "_last_health_ok", True)

    prior_turns = [
        {"role": "user", "content": [{"type": "text", "text": "First question."}]},
        {"role": "assistant", "content": [{"type": "text", "text": "First answer."}]},
    ]

    result = service.run(
        user_prompt="Second question",
        attachment=TurnAttachment(),
        prior_turns=prior_turns,
    )

    messages = captured["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1:3] == prior_turns
    assert messages[3]["role"] == "user"
    assert any(p["text"] == "Second question" for p in messages[3]["content"] if p["type"] == "text")
    assert result.response_text == "Test answer"


def test_image_attachment_is_included_in_current_turn(monkeypatch, tmp_path) -> None:
    service = SandboxService(AppConfig())
    captured: dict = {}

    def fake_generate(messages, **kwargs):
        captured["messages"] = messages
        return _fake_generate(messages, **kwargs)

    monkeypatch.setattr(service._client, "generate", fake_generate)
    monkeypatch.setattr(service, "_last_health_ok", True)

    image_file = tmp_path / "photo.png"
    image_file.write_bytes(b"fake-png")

    attachment = TurnAttachment(image_paths=[image_file])
    service.run(
        user_prompt="What is in this image?",
        attachment=attachment,
        prior_turns=[],
    )

    user_turn = captured["messages"][-1]
    content_types = [p["type"] for p in user_turn["content"]]
    assert "image" in content_types
    assert "text" in content_types


def test_system_prompt_from_config_is_used(monkeypatch) -> None:
    config = AppConfig(system_prompt="You are a storyboard director.")
    service = SandboxService(config)
    captured: dict = {}

    def fake_generate(messages, **kwargs):
        captured["messages"] = messages
        return _fake_generate(messages, **kwargs)

    monkeypatch.setattr(service._client, "generate", fake_generate)
    monkeypatch.setattr(service, "_last_health_ok", True)

    service.run(user_prompt="Plan a scene.", attachment=TurnAttachment(), prior_turns=[])

    system_msg = captured["messages"][0]
    assert system_msg["role"] == "system"
    assert "storyboard director" in system_msg["content"][0]["text"]