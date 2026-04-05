from model_serving.planning.benchmark_targets import benchmark_listing_rewrite
from model_serving.planning.benchmarking import BenchmarkScenario
from model_serving.domain import RequestMode


class FakeGateway:
    def __init__(self) -> None:
        self.requests = []

    def rewrite_listing(self, request):
        self.requests.append(request)
        return {
            "title": "Rewritten title",
            "description": "Rewritten description",
        }


def test_benchmark_listing_rewrite_uses_scenario_metadata(monkeypatch) -> None:
    fake_gateway = FakeGateway()

    def fake_get_gateway(*, model_id: str, max_new_tokens: int, enable_thinking: bool):
        assert model_id == "google/gemma-4-E2B-it"
        assert max_new_tokens == 128
        assert enable_thinking is False
        return fake_gateway

    monkeypatch.setattr(
        "model_serving.planning.benchmark_targets._get_gateway",
        fake_get_gateway,
    )

    scenario = BenchmarkScenario(
        name="rewrite",
        model_label="Gemma 4 E2B",
        request_mode=RequestMode.TEXT_ONLY,
        metadata={
            "model_id": "google/gemma-4-E2B-it",
            "title": "old title",
            "description": "old description",
            "marketplace": "ebay",
            "category_hint": "jewelry",
            "max_new_tokens": 128,
            "enable_thinking": False,
        },
    )

    result = benchmark_listing_rewrite(scenario)

    assert result["title"] == "Rewritten title"
    assert fake_gateway.requests[0].title == "old title"
    assert fake_gateway.requests[0].category_hint == "jewelry"


def test_benchmark_listing_rewrite_derives_model_id_from_label(monkeypatch) -> None:
    fake_gateway = FakeGateway()

    def fake_get_gateway(*, model_id: str, max_new_tokens: int, enable_thinking: bool):
        assert model_id == "google/gemma-4-E4B-it"
        return fake_gateway

    monkeypatch.setattr(
        "model_serving.planning.benchmark_targets._get_gateway",
        fake_get_gateway,
    )

    scenario = BenchmarkScenario(
        name="rewrite",
        model_label="Gemma 4 E4B",
        request_mode=RequestMode.TEXT_ONLY,
        metadata={
            "title": "old title",
            "description": "old description",
        },
    )

    benchmark_listing_rewrite(scenario)

    assert fake_gateway.requests[0].marketplace == "ebay"
