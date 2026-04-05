from __future__ import annotations

import torch

from model_serving.config import ServingConfig, GenerationSettings
from model_serving import model_service as model_service_module
from model_serving.model_service import ModelService


class FakeInputs(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    def __call__(self, text, add_special_tokens, return_tensors):
        assert add_special_tokens is False
        assert return_tensors == "pt"
        token_count = max(1, len(text.split()))
        return {"input_ids": torch.ones((1, token_count), dtype=torch.long)}


class FakeProcessor:
    tokenizer = FakeTokenizer()

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is False
        assert add_generation_prompt is True
        assert enable_thinking is False
        return "formatted prompt"

    def __call__(self, text, return_tensors):
        assert text == "formatted prompt"
        assert return_tensors == "pt"
        return FakeInputs({"input_ids": torch.tensor([[1, 2, 3]])})

    def parse_response(self, response):
        return {"response": response.strip()}

    def decode(self, token_ids, skip_special_tokens=False):
        return "Decoded one shot"


class FakeModel:
    def __init__(self) -> None:
        self._parameter = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        yield self._parameter

    def generate(self, **kwargs):
        streamer = kwargs["streamer"]
        streamer.push("Hello")
        streamer.push(" world")
        streamer.finish()


class FakeStreamer:
    def __init__(self, processor, skip_prompt, skip_special_tokens):
        self._chunks: list[str] = []
        self._finished = False

    def push(self, text: str) -> None:
        self._chunks.append(text)

    def finish(self) -> None:
        self._finished = True

    def __iter__(self):
        index = 0
        while not self._finished or index < len(self._chunks):
            while index < len(self._chunks):
                yield self._chunks[index]
                index += 1


class TrackingProcessor:
    def __init__(self, expected_messages):
        self.expected_messages = expected_messages
        self.parse_calls: list[str] = []
        self.tokenizer = FakeTokenizer()

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert messages == self.expected_messages
        assert tokenize is False
        assert add_generation_prompt is True
        assert enable_thinking is False
        return "formatted prompt"

    def __call__(self, text, return_tensors):
        assert text == "formatted prompt"
        assert return_tensors == "pt"
        return FakeInputs({"input_ids": torch.tensor([[11, 22, 33]])})

    def parse_response(self, response):
        self.parse_calls.append(response)
        return {"response": response.strip()}

    def decode(self, token_ids, skip_special_tokens=False):
        return "Decoded one shot"


class TrackingModel(FakeModel):
    def __init__(self) -> None:
        super().__init__()
        self.eval_called = False
        self.generate_kwargs = None

    def eval(self):
        self.eval_called = True

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        if "streamer" in kwargs:
            super().generate(**kwargs)
            return None
        input_ids = kwargs["input_ids"]
        extra = torch.tensor([[44, 55]], dtype=input_ids.dtype)
        return torch.cat((input_ids, extra), dim=1)


def test_generate_simple_text(monkeypatch) -> None:
    service = ModelService(ServingConfig(generation=GenerationSettings(max_new_tokens=64)))
    fake_processor = FakeProcessor()
    fake_model = FakeModel()
    partial_tokens: list[str] = []
    progress_stages: list[str] = []

    monkeypatch.setattr(
        service,
        "_get_text_runtime",
        lambda progress_callback=None: (fake_processor, fake_model),
    )
    monkeypatch.setattr(model_service_module, "TextIteratorStreamer", FakeStreamer)

    response = service.generate(
        messages=[
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": [{"type": "text", "text": "Say hello."}]},
        ],
        progress_callback=lambda stage, _progress, _message: progress_stages.append(stage),
        token_callback=partial_tokens.append,
    )

    assert response["text"] == "Hello world"
    assert response["input_token_count"] == 3
    assert response["output_token_count"] is not None
    assert response["total_token_count"] is not None
    assert response["metadata"]["prompt_char_count"] == len("formatted prompt")
    assert response["metadata"]["response_char_count"] == len("Hello world")
    assert response["metadata"]["timings"]["generation_seconds"] >= 0
    assert "memory" in response["metadata"]
    assert partial_tokens == ["Hello", "Hello world"]
    assert "prepare" in progress_stages
    assert "generate" in progress_stages
    assert "stream" in progress_stages
    assert progress_stages[-1] == "complete"


def test_generate_text_matches_gemma_getting_started_flow(monkeypatch) -> None:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a short joke about saving RAM."},
    ]
    service = ModelService(ServingConfig(model_id="google/gemma-4-E2B-it", generation=GenerationSettings(max_new_tokens=1024)))
    tracking_processor = TrackingProcessor(messages)
    tracking_model = TrackingModel()
    processor_load_calls: list[str] = []
    model_load_calls: list[tuple[str, dict[str, object]]] = []

    # Populate the lazy-import module-level symbols with fakes so the
    # full processor → model → generate path exercises real code.
    # _import_transformers() would fail on transformers 4.57.x (no
    # AutoModelForMultimodalLM), so we short-circuit it.
    monkeypatch.setattr(model_service_module, "_transformers_imported", True)

    monkeypatch.setattr(model_service_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        model_service_module, "AutoProcessor",
        type("FakeAP", (), {"from_pretrained": staticmethod(
            lambda model_id, **kw: processor_load_calls.append(model_id) or tracking_processor
        )}),
    )
    monkeypatch.setattr(
        model_service_module, "AutoModelForMultimodalLM",
        type("FakeAMML", (), {"from_pretrained": staticmethod(
            lambda model_id, **kwargs: model_load_calls.append((model_id, kwargs)) or tracking_model
        )}),
    )
    monkeypatch.setattr(model_service_module, "TextIteratorStreamer", FakeStreamer)

    response = service.generate(messages)

    assert response["text"] == "Hello world"
    assert processor_load_calls == ["google/gemma-4-E2B-it"]
    assert len(model_load_calls) == 1
    assert model_load_calls[0][0] == "google/gemma-4-E2B-it"
    assert model_load_calls[0][1]["low_cpu_mem_usage"] is True
    assert tracking_model.eval_called is True  # _apply_model_optimizations calls eval()
    assert tracking_model.generate_kwargs is not None
    assert tracking_model.generate_kwargs["temperature"] == 1.0
    assert tracking_model.generate_kwargs["top_p"] == 0.95
    assert tracking_model.generate_kwargs["top_k"] == 64
    assert tracking_model.generate_kwargs["max_new_tokens"] == 1024
    assert tracking_processor.parse_calls == ["Hello world"]
    assert response["input_token_count"] == 3
    assert response["output_token_count"] is not None
    assert response["total_token_count"] is not None
    assert response["metadata"]["prompt_char_count"] == len("formatted prompt")
    assert response["metadata"]["response_char_count"] == len("Hello world")
    assert response["metadata"]["timings"]["runtime_load_seconds"] >= 0
    assert "memory" in response["metadata"]


def test_generate_text_one_shot_skips_streamer(monkeypatch) -> None:
    service = ModelService(ServingConfig(generation=GenerationSettings(max_new_tokens=64, stream_output=False)))
    fake_processor = FakeProcessor()
    fake_model = TrackingModel()

    monkeypatch.setattr(
        service,
        "_get_text_runtime",
        lambda progress_callback=None: (fake_processor, fake_model),
    )

    response = service.generate(
        messages=[
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": [{"type": "text", "text": "Say hello."}]},
        ],
    )

    assert response["text"] == "Decoded one shot"
    assert fake_model.generate_kwargs is not None
    assert "streamer" not in fake_model.generate_kwargs
    assert response["input_token_count"] == 3
    assert response["output_token_count"] is not None