"""Persist Streamlit uploaded files to temporary storage."""

from __future__ import annotations

import tempfile
from pathlib import Path


def persist_upload(uploaded_file) -> Path:
    """Write a Streamlit ``UploadedFile`` to a temp file and return its path."""
    suffix = Path(uploaded_file.name).suffix or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(uploaded_file.getbuffer())
        return Path(handle.name)
