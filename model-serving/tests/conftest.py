from __future__ import annotations

from pathlib import Path
import sys

SERVING_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = SERVING_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
