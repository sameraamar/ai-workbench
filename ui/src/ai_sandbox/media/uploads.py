"""Persist Streamlit uploaded files to the shared media directory.

All uploads land in ``SHARED_MEDIA_DIR`` (configured in ui/.env).
This folder is accessible from both Windows (this process) and WSL2 /
vLLM via the corresponding /mnt/c/… path, so the model can read files
directly using file:// URIs instead of receiving base64 payloads.
"""

from __future__ import annotations

import uuid
from pathlib import Path


def persist_upload(uploaded_file) -> Path:
    """Write a Streamlit ``UploadedFile`` to the shared media dir and return its path."""
    from ai_sandbox.config import SHARED_MEDIA_DIR  # lazy import avoids circular deps

    SHARED_MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded_file.name).suffix or ".bin"
    dest = SHARED_MEDIA_DIR / f"{uuid.uuid4().hex}{suffix}"
    dest.write_bytes(uploaded_file.getbuffer())
    return dest
