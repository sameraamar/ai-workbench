import pytest

from gemma_serving import (
    TrafficProfile,
    estimate_concurrent_requests,
    estimate_cost_per_request,
    estimate_required_workers,
    estimate_worker_throughput,
)


def test_traffic_profile_exposes_concurrency() -> None:
    profile = TrafficProfile(registered_users=100, active_request_rate=0.1)

    assert profile.concurrent_requests == 10


def test_estimate_concurrent_requests() -> None:
    assert estimate_concurrent_requests(100, 0.2) == 20


def test_estimate_worker_throughput() -> None:
    assert estimate_worker_throughput(5) == 0.2


def test_estimate_required_workers_rounds_up() -> None:
    assert estimate_required_workers(10.1) == 11


def test_estimate_cost_per_request() -> None:
    assert estimate_cost_per_request(600, 150, 15000) == pytest.approx(0.05)


@pytest.mark.parametrize(
    ("registered_users", "active_request_rate"),
    [(-1, 0.1), (100, -0.1), (100, 1.1)],
)
def test_estimate_concurrent_requests_validates_inputs(
    registered_users: int,
    active_request_rate: float,
) -> None:
    with pytest.raises(ValueError):
        estimate_concurrent_requests(registered_users, active_request_rate)


def test_estimate_worker_throughput_validates_latency() -> None:
    with pytest.raises(ValueError):
        estimate_worker_throughput(0)


def test_estimate_required_workers_validates_inputs() -> None:
    with pytest.raises(ValueError):
        estimate_required_workers(-1)

    with pytest.raises(ValueError):
        estimate_required_workers(1, per_worker_concurrency=0)


def test_estimate_cost_per_request_validates_inputs() -> None:
    with pytest.raises(ValueError):
        estimate_cost_per_request(-1, 0, 1)

    with pytest.raises(ValueError):
        estimate_cost_per_request(0, -1, 1)

    with pytest.raises(ValueError):
        estimate_cost_per_request(0, 0, 0)