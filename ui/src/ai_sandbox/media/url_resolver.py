"""URL resolver for media attachments.

Classifies a URL by media type (image, audio, video, or unknown) based on
the file extension, and downloads non-image URLs to temporary files so they
can be processed by the appropriate media processor.

Image URLs are passed through directly — vLLM and the OpenAI API support
``image_url`` content parts natively, so the server downloads those itself.

Audio and video URLs must be downloaded client-side because:
  - The OpenAI API spec has no ``audio_url`` or ``video_url`` content type
  - Video must be frame-extracted before sending
  - Audio must be decoded into a format the model processor accepts
"""

from __future__ import annotations

import enum
import logging
import tempfile
from pathlib import Path

import httpx

from ai_sandbox.media.image_processor import IMAGE_EXTENSIONS
from ai_sandbox.media.audio_processor import AUDIO_EXTENSIONS
from ai_sandbox.media.video_processor import VIDEO_EXTENSIONS

LOGGER = logging.getLogger(__name__)


class MediaType(enum.Enum):
    """Classification of a URL by its media type."""

    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    UNKNOWN = "unknown"


def classify_url(url: str) -> tuple[MediaType, str]:
    """Determine the media type of a URL from its file extension.

    Returns ``(MediaType, extension)`` where extension includes the dot.
    Query parameters and fragments are stripped before checking.
    """
    clean = url.lower().split("?")[0].split("#")[0]
    ext = Path(clean).suffix
    if ext in IMAGE_EXTENSIONS:
        return MediaType.IMAGE, ext
    if ext in AUDIO_EXTENSIONS:
        return MediaType.AUDIO, ext
    if ext in VIDEO_EXTENSIONS:
        return MediaType.VIDEO, ext
    return MediaType.UNKNOWN, ext


def download_to_temp(url: str, suffix: str, timeout: int = 120) -> Path:
    """Download a URL to a temporary file and return its path.

    Raises ``httpx.HTTPStatusError`` on non-2xx responses.
    """
    LOGGER.info("Downloading %s ...", url[:100])
    resp = httpx.get(url, follow_redirects=True, timeout=timeout)
    resp.raise_for_status()
    tmp = Path(tempfile.mktemp(suffix=suffix))
    tmp.write_bytes(resp.content)
    size_kb = len(resp.content) / 1024
    LOGGER.info("Downloaded %.0f KB to %s", size_kb, tmp)
    return tmp


def resolve_media_url(url: str) -> tuple[MediaType, Path | str]:
    """Classify and optionally download a media URL.

    Returns:
        ``(MediaType.IMAGE, url)`` — image URLs are passed through as-is
        ``(MediaType.AUDIO, temp_path)`` — audio downloaded to temp file
        ``(MediaType.VIDEO, temp_path)`` — video downloaded to temp file
        ``(MediaType.UNKNOWN, url)`` — unrecognized URL returned unchanged

    Raises ``httpx.HTTPStatusError`` if download fails.
    """
    media_type, ext = classify_url(url)

    if media_type == MediaType.IMAGE:
        # Images pass through — the server downloads them via image_url
        return MediaType.IMAGE, url

    if media_type == MediaType.AUDIO:
        path = download_to_temp(url, suffix=ext, timeout=60)
        return MediaType.AUDIO, path

    if media_type == MediaType.VIDEO:
        path = download_to_temp(url, suffix=ext, timeout=120)
        return MediaType.VIDEO, path

    return MediaType.UNKNOWN, url
