from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    PACKAGE_ROOT = Path(__file__).resolve().parents[1]
    MODEL_SERVING_SRC = PACKAGE_ROOT / "model-serving" / "src"
    if str(MODEL_SERVING_SRC) not in sys.path:
        sys.path.insert(0, str(MODEL_SERVING_SRC))
    from gemma_serving.simulation import simulate_capacity
else:
    from .simulation import simulate_capacity


def main() -> int:
    parser = argparse.ArgumentParser(description="Simulate E2B vs E4B serving capacity.")
    parser.add_argument("--registered-users", type=int, default=100)
    parser.add_argument("--active-request-rate", type=float, default=0.1)
    parser.add_argument("--multimodal-share", type=float, default=0.2)
    parser.add_argument("--monthly-successful-requests", type=int)
    args = parser.parse_args()

    snapshots = simulate_capacity(
        registered_users=args.registered_users,
        active_request_rate=args.active_request_rate,
        multimodal_share=args.multimodal_share,
        monthly_successful_requests=args.monthly_successful_requests,
    )
    print(json.dumps([snapshot.to_dict() for snapshot in snapshots], indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())