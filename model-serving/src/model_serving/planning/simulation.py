from __future__ import annotations

from dataclasses import asdict, dataclass

from ..domain import RequestMode
from .planning import (
    estimate_concurrent_requests,
    estimate_cost_per_request,
    estimate_required_workers,
    estimate_worker_throughput,
)


@dataclass(frozen=True)
class ModelProfile:
    model_label: str
    text_latency_seconds: float
    multimodal_latency_seconds: float
    monthly_worker_cost: float = 0.0
    monthly_infra_overhead: float = 0.0


@dataclass(frozen=True)
class CapacitySnapshot:
    model_label: str
    registered_users: int
    active_request_rate: float
    multimodal_share: float
    concurrent_requests: float
    blended_latency_seconds: float
    worker_throughput_per_second: float
    worker_throughput_per_minute: float
    required_workers: int
    estimated_monthly_cost_per_request: float | None

    def to_dict(self) -> dict[str, float | int | str | None]:
        return asdict(self)


MODEL_PROFILES: dict[str, ModelProfile] = {
    "E2B": ModelProfile(
        model_label="Gemma 4 E2B",
        text_latency_seconds=2.5,
        multimodal_latency_seconds=5.5,
    ),
    "E4B": ModelProfile(
        model_label="Gemma 4 E4B",
        text_latency_seconds=4.0,
        multimodal_latency_seconds=8.0,
    ),
}


def simulate_capacity(
    *,
    registered_users: int,
    active_request_rate: float,
    multimodal_share: float,
    monthly_successful_requests: int | None = None,
    profiles: dict[str, ModelProfile] | None = None,
) -> list[CapacitySnapshot]:
    if not 0 <= multimodal_share <= 1:
        raise ValueError("multimodal_share must be between 0 and 1")

    active_profiles = profiles or MODEL_PROFILES
    concurrent_requests = estimate_concurrent_requests(registered_users, active_request_rate)
    snapshots: list[CapacitySnapshot] = []

    for profile in active_profiles.values():
        blended_latency = _blended_latency(profile, multimodal_share)
        throughput_per_second = estimate_worker_throughput(blended_latency)
        required_workers = estimate_required_workers(concurrent_requests)
        monthly_cost_per_request = None
        if monthly_successful_requests is not None:
            monthly_cost_per_request = estimate_cost_per_request(
                monthly_gpu_cost=profile.monthly_worker_cost * required_workers,
                monthly_infra_overhead=profile.monthly_infra_overhead * required_workers,
                monthly_successful_requests=monthly_successful_requests,
            )

        snapshots.append(
            CapacitySnapshot(
                model_label=profile.model_label,
                registered_users=registered_users,
                active_request_rate=active_request_rate,
                multimodal_share=multimodal_share,
                concurrent_requests=concurrent_requests,
                blended_latency_seconds=blended_latency,
                worker_throughput_per_second=throughput_per_second,
                worker_throughput_per_minute=throughput_per_second * 60,
                required_workers=required_workers,
                estimated_monthly_cost_per_request=monthly_cost_per_request,
            )
        )

    return snapshots


def _blended_latency(profile: ModelProfile, multimodal_share: float) -> float:
    return (profile.text_latency_seconds * (1 - multimodal_share)) + (
        profile.multimodal_latency_seconds * multimodal_share
    )
