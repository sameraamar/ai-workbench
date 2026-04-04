from __future__ import annotations

from pathlib import Path
import os
import sys

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent


def bootstrap_environment() -> None:
    load_dotenv(ROOT / ".env", override=False)
    _apply_pythonpath_from_env()


def _apply_pythonpath_from_env() -> None:
    for env_name in ("APP_PYTHONPATH", "PYTHONPATH"):
        raw_value = os.getenv(env_name, "").strip()
        if not raw_value:
            continue
        for segment in raw_value.split(os.pathsep):
            clean_segment = segment.strip()
            if not clean_segment:
                continue
            candidate = Path(clean_segment)
            if not candidate.is_absolute():
                candidate = ROOT / candidate
            resolved = str(candidate.resolve())
            if resolved not in sys.path:
                sys.path.insert(0, resolved)
