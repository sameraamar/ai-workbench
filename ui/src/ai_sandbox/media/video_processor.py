"""Video preprocessing.

Gemma 4 (and most multimodal LLMs) cannot process raw video. Video must be
converted to a sequence of representative image frames before being sent to
the model. This processor handles:

1. Opening the video file via OpenCV
2. Sampling a fixed number of evenly-spaced frames
3. Saving each frame as a temporary PNG

The thumbnail is generated from the first frame of the video.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import cv2

from ai_sandbox.media.base import MediaProcessor, _image_to_data_uri

LOGGER = logging.getLogger(__name__)

VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpg", ".mpeg"})

# Default number of frames to extract from a video.
DEFAULT_MAX_FRAMES = 6


class VideoProcessor(MediaProcessor):
    """Classify, preview, and prepare video files by extracting frames."""

    extensions = VIDEO_EXTENSIONS

    @classmethod
    def is_video(cls, path: Path) -> bool:
        """Return True if the file extension indicates a video file."""
        return cls.matches(path)

    @classmethod
    def make_thumbnail_data_uri(cls, path: Path, max_size: int = 80) -> str | None:
        """Extract the first frame and return it as a small PNG data URI."""
        try:
            from PIL import Image as _PILImage
            import numpy as np

            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                return None
            success, frame = capture.read()
            capture.release()
            if not success:
                return None
            # OpenCV uses BGR; convert to RGB for PIL
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = _PILImage.fromarray(rgb)
            return _image_to_data_uri(img, max_size)
        except Exception:
            LOGGER.debug("Failed to create video thumbnail for %s", path, exc_info=True)
            return None

    @classmethod
    def preprocess(cls, path: Path, **kwargs: Any) -> dict[str, Any]:
        """Extract frames from the video.

        Keyword args:
            max_frames (int): Number of frames to extract (default: 6).

        Returns ``{"frame_paths": [path, ...]}``.
        """
        max_frames = kwargs.get("max_frames", DEFAULT_MAX_FRAMES)
        frames = extract_video_frames(path, max_frames=max_frames)
        return {"frame_paths": frames}


# ---------------------------------------------------------------------------
# Standalone function (used by __init__.py for backward compat)
# ---------------------------------------------------------------------------

def extract_video_frames(video_path: Path, max_frames: int = DEFAULT_MAX_FRAMES) -> list[Path]:
    """Extract evenly-spaced frames from a video file.

    Returns a list of paths to temporary PNG files.
    Raises ``ValueError`` if the video cannot be opened or contains no frames.
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("Unable to open video file for frame extraction.")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        capture.release()
        raise ValueError("Video file does not contain readable frames.")

    target_indexes = sorted(
        {int(idx) for idx in _linspace_indexes(frame_count, max_frames)}
    )
    output_paths: list[Path] = []

    for frame_index in target_indexes:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = capture.read()
        if not success:
            continue
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as handle:
            cv2.imwrite(handle.name, frame)
            output_paths.append(Path(handle.name))

    capture.release()

    if not output_paths:
        raise ValueError("No frames could be extracted from the video.")

    LOGGER.info(
        "Extracted %d frames from %s (total %d in video)",
        len(output_paths), video_path.name, frame_count,
    )
    return output_paths


def _linspace_indexes(frame_count: int, samples: int) -> list[int]:
    """Return *samples* evenly-spaced frame indexes."""
    if samples <= 1 or frame_count <= 1:
        return [0]
    if samples >= frame_count:
        return list(range(frame_count))
    last = frame_count - 1
    return [round(last * i / (samples - 1)) for i in range(samples)]
