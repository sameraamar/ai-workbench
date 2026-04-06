"""Media preprocessing package.

Provides processors for each media type (image, audio, video) and a URL
resolver that downloads remote media files when needed.

Usage:
    from ai_sandbox.media import (
        persist_upload,
        ImageProcessor,
        AudioProcessor,
        VideoProcessor,
        resolve_media_url,
    )
"""

from ai_sandbox.media.base import MediaProcessor
from ai_sandbox.media.uploads import persist_upload
from ai_sandbox.media.image_processor import ImageProcessor
from ai_sandbox.media.audio_processor import AudioProcessor
from ai_sandbox.media.video_processor import VideoProcessor
from ai_sandbox.media.url_resolver import resolve_media_url, MediaType

# Backward compat: extract_video_frames was used directly in app.py
from ai_sandbox.media.video_processor import extract_video_frames

__all__ = [
    "MediaProcessor",
    "persist_upload",
    "ImageProcessor",
    "AudioProcessor",
    "VideoProcessor",
    "resolve_media_url",
    "MediaType",
    "extract_video_frames",
]
