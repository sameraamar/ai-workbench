from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path
import statistics
import time
from typing import Any, Callable

from ..domain import RequestMode

BenchmarkTarget = Callable[["BenchmarkScenario"], Any]
Clock = Callable[[], float]
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BenchmarkScenario:
    name: str
    model_label: str
    request_mode: RequestMode
    iterations: int = 5
    warmup_iterations: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if self.warmup_iterations < 0:
            raise ValueError("warmup_iterations must be non-negative")


@dataclass(frozen=True)
class BenchmarkSummary:
    scenario_name: str
    model_label: str
    request_mode: RequestMode
    warmup_iterations: int
    measured_iterations: int
    average_latency_seconds: float
    p50_latency_seconds: float
    p95_latency_seconds: float
    requests_per_minute: float
    latencies_seconds: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["request_mode"] = self.request_mode.value
        return payload


def run_benchmark(
    target: BenchmarkTarget,
    scenario: BenchmarkScenario,
    *,
    clock: Clock = time.perf_counter,
) -> BenchmarkSummary:
    LOGGER.info(
        "Starting benchmark scenario '%s' for %s (%s)",
        scenario.name,
        scenario.model_label,
        scenario.request_mode.value,
    )
    for _ in range(scenario.warmup_iterations):
        LOGGER.info("Warmup run for scenario '%s'", scenario.name)
        target(scenario)

    latencies: list[float] = []
    for iteration in range(1, scenario.iterations + 1):
        LOGGER.info(
            "Measured iteration %s/%s for scenario '%s'",
            iteration,
            scenario.iterations,
            scenario.name,
        )
        started = clock()
        target(scenario)
        completed = clock()
        latency = completed - started
        latencies.append(latency)
        LOGGER.info(
            "Iteration %s/%s finished in %.3fs",
            iteration,
            scenario.iterations,
            latency,
        )

    average_latency = statistics.fmean(latencies)
    LOGGER.info(
        "Completed scenario '%s': avg=%.3fs p50=%.3fs p95=%.3fs rpm=%.2f",
        scenario.name,
        average_latency,
        _percentile(latencies, 0.50),
        _percentile(latencies, 0.95),
        60 / average_latency if average_latency else 0.0,
    )
    return BenchmarkSummary(
        scenario_name=scenario.name,
        model_label=scenario.model_label,
        request_mode=scenario.request_mode,
        warmup_iterations=scenario.warmup_iterations,
        measured_iterations=scenario.iterations,
        average_latency_seconds=average_latency,
        p50_latency_seconds=_percentile(latencies, 0.50),
        p95_latency_seconds=_percentile(latencies, 0.95),
        requests_per_minute=60 / average_latency if average_latency else 0.0,
        latencies_seconds=tuple(latencies),
    )


def load_scenarios(file_path: Path) -> list[BenchmarkScenario]:
    raw_payload = json.loads(file_path.read_text(encoding="utf-8"))
    scenarios: list[BenchmarkScenario] = []
    for item in raw_payload:
        scenarios.append(
            BenchmarkScenario(
                name=item["name"],
                model_label=item["model_label"],
                request_mode=RequestMode(item["request_mode"]),
                iterations=item.get("iterations", 5),
                warmup_iterations=item.get("warmup_iterations", 1),
                metadata=item.get("metadata", {}),
            )
        )
    return scenarios


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, round((len(ordered) - 1) * fraction))
    return ordered[index]
