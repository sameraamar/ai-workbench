"""Base class for media processors.

All media processors share the same interface so consumer code can
treat them uniformly. Each processor handles one media type (image,
audio, or video) and provides:

- ``extensions``  — recognized file extensions
- ``matches(path)``  — classify a file by extension
- ``make_thumbnail_data_uri(path)``  — small preview for chat history
- ``preprocess(path)``  — prepare media for the model (returns paths/URLs)
"""

from __future__ import annotations

import base64
import logging
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


class MediaProcessor(ABC):
    """Interface that all media processors must implement."""

    #: File extensions this processor handles (including the dot).
    extensions: frozenset[str] = frozenset()

    @classmethod
    def matches(cls, path: Path) -> bool:
        """Return True if *path*'s extension belongs to this media type."""
        return path.suffix.lower() in cls.extensions

    @classmethod
    @abstractmethod
    def make_thumbnail_data_uri(cls, path: Path, max_size: int = 80) -> str | None:
        """Return a small base64 PNG data URI for display in chat history.

        Returns ``None`` if a preview cannot be generated.
        """

    @classmethod
    @abstractmethod
    def preprocess(cls, path: Path, **kwargs: Any) -> dict[str, Any]:
        """Prepare the media file for sending to the model.

        Returns a dict with processor-specific keys, e.g.:
          - ImageProcessor: ``{"image_paths": [path]}``
          - AudioProcessor: ``{"audio_path": path}``
          - VideoProcessor: ``{"frame_paths": [path, ...]}``
        """


def _image_to_data_uri(img, max_size: int = 80) -> str:
    """Resize a PIL Image and encode as a base64 PNG data URI."""
    img.thumbnail((max_size, max_size))
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"
