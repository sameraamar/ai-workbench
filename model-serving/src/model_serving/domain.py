from dataclasses import dataclass
from enum import Enum


class RequestMode(str, Enum):
    TEXT_ONLY = "text-only"
    MULTIMODAL = "multimodal"


@dataclass(frozen=True)
class TrafficProfile:
    registered_users: int
    active_request_rate: float

    def __post_init__(self) -> None:
        if self.registered_users < 0:
            raise ValueError("registered_users must be non-negative")
        if not 0 <= self.active_request_rate <= 1:
            raise ValueError("active_request_rate must be between 0 and 1")

    @property
    def concurrent_requests(self) -> float:
        return self.registered_users * self.active_request_rate