from .config import GenerationSettings, ServingConfig
from .domain import RequestMode, TrafficProfile
from .planning import (
    estimate_concurrent_requests,
    estimate_cost_per_request,
    estimate_required_workers,
    estimate_worker_throughput,
)
from .simulation import MODEL_PROFILES, simulate_capacity

__all__ = [
    "GenerationSettings",
    "GemmaLowCostGateway",
    "LowCostServingConfig",
    "MODEL_PROFILES",
    "RequestMode",
    "ServingConfig",
    "StubLowCostGateway",
    "TrafficProfile",
    "create_low_cost_app",
    "estimate_concurrent_requests",
    "estimate_cost_per_request",
    "estimate_required_workers",
    "estimate_worker_throughput",
    "simulate_capacity",
]


def __getattr__(name: str):
    if name == "GemmaLowCostGateway":
        from .gateway import GemmaLowCostGateway

        return GemmaLowCostGateway
    if name in {"LowCostServingConfig", "StubLowCostGateway", "create_low_cost_app"}:
        from .app import LowCostServingConfig, StubLowCostGateway, create_low_cost_app

        return {
            "LowCostServingConfig": LowCostServingConfig,
            "StubLowCostGateway": StubLowCostGateway,
            "create_low_cost_app": create_low_cost_app,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
