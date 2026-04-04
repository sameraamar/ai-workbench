from pathlib import Path

import pytest

from gemma_serving.benchmarking import BenchmarkScenario, load_scenarios, run_benchmark
from gemma_serving.domain import RequestMode
from gemma_serving.simulation import simulate_capacity


def test_run_benchmark_returns_summary() -> None:
    scenario = BenchmarkScenario(
        name="rewrite",
        model_label="Gemma 4 E2B",
        request_mode=RequestMode.TEXT_ONLY,
        iterations=3,
        warmup_iterations=1,
    )
    clock_values = iter([0.0, 0.1, 0.2, 0.5, 0.7, 1.1])

    def fake_clock() -> float:
        return next(clock_values)

    def fake_target(_: BenchmarkScenario) -> dict[str, str]:
        return {"status": "ok"}

    summary = run_benchmark(fake_target, scenario, clock=fake_clock)

    assert summary.measured_iterations == 3
    assert summary.p50_latency_seconds == pytest.approx(0.3)
    assert summary.p95_latency_seconds == pytest.approx(0.4)


def test_load_scenarios_parses_request_mode(tmp_path: Path) -> None:
    scenario_file = tmp_path / "scenarios.json"
    scenario_file.write_text(
        "["
        '{"name": "rewrite", "model_label": "Gemma 4 E2B", "request_mode": "text-only"}'
        "]",
        encoding="utf-8",
    )

    scenarios = load_scenarios(scenario_file)

    assert scenarios[0].request_mode is RequestMode.TEXT_ONLY


def test_simulate_capacity_returns_e2b_and_e4b() -> None:
    snapshots = simulate_capacity(
        registered_users=100,
        active_request_rate=0.1,
        multimodal_share=0.2,
    )

    labels = {snapshot.model_label for snapshot in snapshots}
    assert labels == {"Gemma 4 E2B", "Gemma 4 E4B"}