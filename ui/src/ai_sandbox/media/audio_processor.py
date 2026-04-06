"""Audio preprocessing.

Audio files are sent as-is to the model (the model's processor handles
decoding internally). The thumbnail is a simple waveform visualization
generated from the first few seconds of audio.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ai_sandbox.media.base import MediaProcessor, _image_to_data_uri

LOGGER = logging.getLogger(__name__)

AUDIO_EXTENSIONS = frozenset({".wav", ".mp3", ".flac", ".ogg", ".m4a"})


class AudioProcessor(MediaProcessor):
    """Classify, preview, and prepare audio files."""

    extensions = AUDIO_EXTENSIONS

    @classmethod
    def is_audio(cls, path: Path) -> bool:
        """Return True if the file extension indicates an audio file."""
        return cls.matches(path)

    @classmethod
    def make_thumbnail_data_uri(cls, path: Path, max_size: int = 80) -> str | None:
        """Generate a small waveform thumbnail from the audio file.

        Uses a simple matplotlib plot of the waveform. Falls back to ``None``
        if soundfile or matplotlib are not installed.
        """
        try:
            import soundfile as sf
            import numpy as np
            from PIL import Image as _PILImage
            from io import BytesIO

            # Read first 3 seconds max
            info = sf.info(str(path))
            max_samples = int(min(info.duration, 3.0) * info.samplerate)
            data, _sr = sf.read(str(path), frames=max_samples, dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)  # mono

            # Draw a simple waveform as a tiny image
            width, height = max_size, max_size // 2
            img = _PILImage.new("RGB", (width, height), color=(40, 40, 50))
            pixels = img.load()

            # Downsample to width points
            step = max(1, len(data) // width)
            samples = data[::step][:width]
            mid = height // 2
            for x, val in enumerate(samples):
                y = int(mid - val * mid * 0.9)
                y = max(0, min(height - 1, y))
                # Draw a vertical line from mid to y
                for dy in range(min(mid, y), max(mid, y) + 1):
                    pixels[x, dy] = (80, 180, 255)

            return _image_to_data_uri(img, max_size)
        except Exception:
            LOGGER.debug("Failed to create audio thumbnail for %s", path, exc_info=True)
            return None

    @classmethod
    def preprocess(cls, path: Path, **kwargs: Any) -> dict[str, Any]:
        """Return the audio path — no transformation needed.

        Returns ``{"audio_path": path}``.
        """
        return {"audio_path": path}
