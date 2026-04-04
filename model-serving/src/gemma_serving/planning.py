import math


def estimate_concurrent_requests(registered_users: int, active_request_rate: float) -> float:
    if registered_users < 0:
        raise ValueError("registered_users must be non-negative")
    if not 0 <= active_request_rate <= 1:
        raise ValueError("active_request_rate must be between 0 and 1")
    return registered_users * active_request_rate


def estimate_worker_throughput(average_request_latency_seconds: float) -> float:
    if average_request_latency_seconds <= 0:
        raise ValueError("average_request_latency_seconds must be positive")
    return 1 / average_request_latency_seconds


def estimate_required_workers(
    target_concurrent_requests: float,
    per_worker_concurrency: int = 1,
) -> int:
    if target_concurrent_requests < 0:
        raise ValueError("target_concurrent_requests must be non-negative")
    if per_worker_concurrency <= 0:
        raise ValueError("per_worker_concurrency must be positive")
    return math.ceil(target_concurrent_requests / per_worker_concurrency)


def estimate_cost_per_request(
    monthly_gpu_cost: float,
    monthly_infra_overhead: float,
    monthly_successful_requests: int,
) -> float:
    if monthly_gpu_cost < 0:
        raise ValueError("monthly_gpu_cost must be non-negative")
    if monthly_infra_overhead < 0:
        raise ValueError("monthly_infra_overhead must be non-negative")
    if monthly_successful_requests <= 0:
        raise ValueError("monthly_successful_requests must be positive")
    return (monthly_gpu_cost + monthly_infra_overhead) / monthly_successful_requests