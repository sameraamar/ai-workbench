"""Tests for the OpenAI-compatible API shim (openai_compat.py).

These tests exercise message format conversion and response building
without requiring a running model or GPU.
"""

from __future__ import annotations

import pytest

from model_serving.openai_compat import (
    ChatMessage,
    _openai_to_internal_messages,
    _chat_completion_response,
)


# ---------------------------------------------------------------------------
# _openai_to_internal_messages
# ---------------------------------------------------------------------------

class TestOpenAIToInternalMessages:
    def test_plain_text_passthrough(self) -> None:
        msgs = [ChatMessage(role="user", content="hello")]
        result = _openai_to_internal_messages(msgs)
        assert result == [{"role": "user", "content": "hello"}]

    def test_system_message_passthrough(self) -> None:
        msgs = [ChatMessage(role="system", content="You are helpful.")]
        result = _openai_to_internal_messages(msgs)
        assert result == [{"role": "system", "content": "You are helpful."}]

    def test_image_url_converted_to_internal_image(self) -> None:
        msgs = [
            ChatMessage(
                role="user",
                content=[
                    {"type": "text", "text": "What is this?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/cat.jpg"},
                    },
                ],
            )
        ]
        result = _openai_to_internal_messages(msgs)
        assert len(result) == 1
        parts = result[0]["content"]
        assert parts[0] == {"type": "text", "text": "What is this?"}
        assert parts[1] == {"type": "image", "url": "https://example.com/cat.jpg"}

    def test_mixed_conversation(self) -> None:
        msgs = [
            ChatMessage(role="system", content="Be brief."),
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assistant", content="Hello!"),
            ChatMessage(
                role="user",
                content=[
                    {"type": "text", "text": "Describe this"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,abc123"},
                    },
                ],
            ),
        ]
        result = _openai_to_internal_messages(msgs)
        assert len(result) == 4
        assert result[0]["content"] == "Be brief."
        assert result[1]["content"] == "Hi"
        assert result[2]["content"] == "Hello!"
        assert result[3]["content"][1]["type"] == "image"

    def test_unknown_block_type_passed_through(self) -> None:
        msgs = [
            ChatMessage(
                role="user",
                content=[
                    {"type": "text", "text": "test"},
                    {"type": "custom_thing", "data": "foo"},
                ],
            )
        ]
        result = _openai_to_internal_messages(msgs)
        assert result[0]["content"][1] == {"type": "custom_thing", "data": "foo"}


# ---------------------------------------------------------------------------
# _chat_completion_response
# ---------------------------------------------------------------------------

class TestChatCompletionResponse:
    def test_basic_response(self) -> None:
        resp = _chat_completion_response(
            text="Hello there!",
            model_id="google/gemma-4-E2B-it",
            prompt_tokens=10,
            completion_tokens=5,
        )
        assert resp["object"] == "chat.completion"
        assert resp["model"] == "google/gemma-4-E2B-it"
        assert resp["choices"][0]["message"]["content"] == "Hello there!"
        assert resp["choices"][0]["finish_reason"] == "stop"
        assert resp["usage"]["prompt_tokens"] == 10
        assert resp["usage"]["completion_tokens"] == 5
        assert resp["usage"]["total_tokens"] == 15

    def test_none_token_counts(self) -> None:
        resp = _chat_completion_response(
            text="test",
            model_id="model",
        )
        assert resp["usage"]["prompt_tokens"] == 0
        assert resp["usage"]["total_tokens"] == 0
