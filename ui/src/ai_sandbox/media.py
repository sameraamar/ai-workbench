from __future__ import annotations

from pathlib import Path
import tempfile

import cv2


def persist_upload(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(uploaded_file.getbuffer())
        return Path(handle.name)


def extract_video_frames(video_path: Path, max_frames: int = 6) -> list[Path]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("Unable to open video file for frame extraction.")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        capture.release()
        raise ValueError("Video file does not contain readable frames.")

    target_indexes = sorted({int(index) for index in _linspace_indexes(frame_count, max_frames)})
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
    return output_paths


def _linspace_indexes(frame_count: int, samples: int) -> list[int]:
    if samples <= 1 or frame_count <= 1:
        return [0]
    if samples >= frame_count:
        return list(range(frame_count))
    last_index = frame_count - 1
    return [round(last_index * step / (samples - 1)) for step in range(samples)]