from __future__ import annotations

import argparse
from importlib import import_module
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Callable

if __package__ in {None, ""}:
    PACKAGE_ROOT = Path(__file__).resolve().parents[1]
    MODEL_SERVING_SRC = PACKAGE_ROOT / "model-serving" / "src"
    if str(MODEL_SERVING_SRC) not in sys.path:
        sys.path.insert(0, str(MODEL_SERVING_SRC))
    from gemma_serving.benchmarking import BenchmarkScenario, load_scenarios, run_benchmark
else:
    from .benchmarking import BenchmarkScenario, load_scenarios, run_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(description="Run serving research benchmarks.")
    parser.add_argument("scenario_file", type=Path, help="Path to a JSON file containing benchmark scenarios.")
    parser.add_argument(
        "--target",
        default="simulate",
        help="Callable in module:function form. Defaults to a built-in latency simulator.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress benchmark progress logs and only print final JSON results.",
    )
    args = parser.parse_args()

    _configure_logging(quiet=args.quiet)
    LOGGER.info("Loading scenarios from %s", args.scenario_file)

    target = _load_target(args.target)
    scenarios = load_scenarios(args.scenario_file)
    LOGGER.info("Loaded %s scenarios using target '%s'", len(scenarios), args.target)
    results = [run_benchmark(target, scenario).to_dict() for scenario in scenarios]
    print(json.dumps(results, indent=2))
    return 0


def _load_target(target_reference: str) -> Callable[[BenchmarkScenario], Any]:
    if target_reference == "simulate":
        LOGGER.info("Using built-in simulated benchmark target")
        return _simulate_target

    module_name, function_name = target_reference.split(":", maxsplit=1)
    LOGGER.info("Loading benchmark target %s from module %s", function_name, module_name)
    module = import_module(module_name)
    return getattr(module, function_name)


def _configure_logging(*, quiet: bool) -> None:
    logging.basicConfig(
        level=logging.WARNING if quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


LOGGER = logging.getLogger(__name__)


def _simulate_target(scenario: BenchmarkScenario) -> dict[str, Any]:
    simulated_latency = float(scenario.metadata.get("simulated_latency_seconds", 0.05))
    time.sleep(simulated_latency)
    return {
        "scenario": scenario.name,
        "model_label": scenario.model_label,
        "simulated_latency_seconds": simulated_latency,
    }


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())