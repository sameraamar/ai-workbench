from __future__ import annotations

import logging
from pathlib import Path
import sys
import tempfile
from time import perf_counter

from env_bootstrap import bootstrap_environment

bootstrap_environment()

import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gemma_sandbox.config import AppConfig, DEFAULT_MODEL_ID, DEFAULT_SYSTEM_PROMPT, GenerationSettings, SERVING_URL
from gemma_sandbox.domain import RunResult
from gemma_sandbox.media import extract_video_frames, persist_upload
from gemma_sandbox.model_profiles import MODEL_LABELS, get_capabilities, get_model_id
from gemma_sandbox.services.sandbox_service import SandboxService, TurnAttachment

try:
    from streamlit_paste_button import paste_image_button as _paste_image_button
    _PASTE_AVAILABLE = True
except ImportError:
    _PASTE_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

LOGGER = logging.getLogger(__name__)


st.set_page_config(
    page_title="AI Sandbox Arena",
    page_icon="AI",
    layout="wide",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;600&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=swap');
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(255, 208, 110, 0.22), transparent 28%),
                radial-gradient(circle at top right, rgba(93, 196, 255, 0.18), transparent 24%),
                linear-gradient(160deg, #f8f3ea 0%, #eef4f7 52%, #f4efe3 100%);
        }
        html, body, [class*="css"] {
            font-family: 'Space Grotesk', sans-serif;
        }
        .mono-label {
            font-family: 'IBM Plex Mono', monospace;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            font-size: 0.78rem;
            color: #4f5d75;
        }
        .hero {
            padding: 1.2rem 1.4rem;
            border: 1px solid rgba(62, 84, 97, 0.12);
            border-radius: 24px;
            background: rgba(255, 255, 255, 0.72);
            box-shadow: 0 12px 30px rgba(82, 88, 102, 0.08);
        }
        .support-chip {
            display: inline-block;
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
            border: 1px solid rgba(56, 67, 88, 0.2);
            background: rgba(255, 255, 255, 0.9);
            font-size: 0.82rem;
            margin-bottom: 0.5rem;
        }
        /* Chat input — visible border, focus ring, and matching design system */
        [data-testid="stChatInput"] {
            border: 1.5px solid rgba(60, 80, 110, 0.30) !important;
            border-radius: 16px !important;
            background: rgba(255, 255, 255, 0.90) !important;
            box-shadow: 0 2px 10px rgba(60, 80, 110, 0.07) !important;
            transition: border-color 0.18s, box-shadow 0.18s;
        }
        [data-testid="stChatInput"]:focus-within {
            border-color: rgba(55, 115, 200, 0.55) !important;
            box-shadow: 0 0 0 3px rgba(55, 115, 200, 0.12), 0 2px 10px rgba(60, 80, 110, 0.07) !important;
        }
        /* Prompt label row with Material icon */
        .prompt-label-row {
            display: flex;
            align-items: center;
            gap: 0.32rem;
            margin-bottom: 0.4rem;
            font-size: 0.88rem;
            font-weight: 500;
            color: #3d4f62;
        }
        .prompt-label-row .mat-icon {
            font-family: 'Material Symbols Rounded';
            font-size: 1.15rem;
            vertical-align: middle;
            line-height: 1;
            color: #6b84a0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def get_sandbox_service(config: AppConfig) -> SandboxService:
    return SandboxService(config)


def _calculate_tokens_per_second(output_token_count: int | None, elapsed_seconds: float) -> float | None:
    if output_token_count is None or elapsed_seconds <= 0:
        return None
    return output_token_count / elapsed_seconds


# File extension sets used to detect what kind of media is attached per turn.
_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".webp"})
_AUDIO_EXTS = frozenset({".wav", ".mp3", ".flac", ".ogg", ".m4a"})
_VIDEO_EXTS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm"})


def _conversation_key(*, model_id: str, system_prompt: str) -> str:
    return f"{model_id}|{system_prompt.strip()}"


def _get_model_history_store() -> dict[str, list[dict]]:
    return st.session_state.setdefault("conversation_model_history", {})


def _get_ui_history_store() -> dict[str, list[dict[str, str]]]:
    return st.session_state.setdefault("conversation_ui_history", {})


def _get_history_for_key(key: str) -> tuple[list[dict], list[dict[str, str]]]:
    model_store = _get_model_history_store()
    ui_store = _get_ui_history_store()
    return model_store.setdefault(key, []), ui_store.setdefault(key, [])


def _clear_history_for_key(key: str) -> None:
    _get_model_history_store().pop(key, None)
    _get_ui_history_store().pop(key, None)


def _render_conversation_history(history: list[dict]) -> None:
    if not history:
        st.caption("Conversation is empty. Send a message to start.")
        return
    for message in history:
        role = message["role"]
        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(message["text"])
            labels = message.get("attachment_labels", [])
            if labels:
                st.caption("📎 " + ", ".join(labels))


def _render_pending_exchange(user_prompt: str, attachment_labels: list[str]) -> st.delta_generator.DeltaGenerator:
    with st.chat_message("user"):
        st.markdown(user_prompt)
        if attachment_labels:
            st.caption("📎 " + ", ".join(attachment_labels))
    with st.chat_message("assistant"):
        placeholder = st.empty()
    return placeholder


def _model_family_label(label: str) -> str:
    """Return a short family name for display from a model dropdown label."""
    low = label.lower()
    if "mistral" in low:
        return "Mistral"
    if "gemma" in low:
        return "Gemma 4"
    return label.split("(")[0].strip()


def main() -> None:
    inject_styles()

    with st.sidebar:
        st.header("Settings")
        selected_model_label = st.selectbox(
            "Model",
            options=MODEL_LABELS,
            index=0,
        )
        model_id = get_model_id(selected_model_label, fallback=DEFAULT_MODEL_ID)

        caps = get_capabilities(selected_model_label)

        # --- Model load control ---
        _server_model_id = st.session_state.get("_loaded_model_id")
        _model_ready = st.session_state.get("_model_ready", False)
        model_changed = model_id != _server_model_id
        if model_changed:
            _model_ready = False
            st.session_state["_model_ready"] = False

        load_col, status_col = st.columns([0.5, 0.5])
        with load_col:
            load_clicked = st.button(
                "Load Model",
                type="primary",
                use_container_width=True,
                disabled=_model_ready and not model_changed,
            )
        with status_col:
            if _model_ready and not model_changed:
                st.success("Ready", icon="\u2705")
            elif _server_model_id and model_changed:
                st.warning("Changed", icon="\u26A0\uFE0F")
            else:
                st.info("Not loaded", icon="\u23F3")

        if caps.vram_gb_bf16 > 0:
            _vram = caps.vram_gb_bf16
            if _vram > 24:
                st.warning(
                    f"⚠️ **{selected_model_label}** requires ~{_vram:.0f} GB VRAM at BF16. "
                    f"Enable `GEMMA_QUANTIZE_4BIT=1` in `model-serving/.env` to run on a 24 GB GPU (~{_vram / 4:.0f} GB with 4-bit).",
                )
            else:
                st.caption(f"~{_vram:.0f} GB VRAM at BF16.")

        if "_system_prompt_input" not in st.session_state:
            st.session_state["_system_prompt_input"] = DEFAULT_SYSTEM_PROMPT
        system_prompt = st.text_area(
            "System prompt",
            height=160,
            key="_system_prompt_input",
        )
        max_new_tokens = st.slider("Max new tokens", min_value=64, max_value=2048, value=256, step=64)
        enable_thinking = st.toggle("Enable thinking", value=False)
        stream_output = st.checkbox("Stream text response to UI", value=True)
        st.caption(f"Active model: {model_id}")
        st.caption(f"Serving URL: {SERVING_URL}")
        st.caption("Standardized sampling defaults are locked to temperature=1.0, top_p=0.95, top_k=64.")

    # Hero — dynamic label reflects the currently selected model family
    _family = _model_family_label(selected_model_label)
    _media_note = "Attach images" + (", audio," if caps.audio else "") + (" video," if caps.video else "") + " or other media to any message. Responses are always text."
    st.markdown('<div class="hero">', unsafe_allow_html=True)
    st.markdown(f'<div class="mono-label">{_family} Sandbox</div>', unsafe_allow_html=True)
    st.title("AI Sandbox Arena")
    st.write(_media_note)
    st.markdown("</div>", unsafe_allow_html=True)

    config = AppConfig(
        model_id=model_id,
        system_prompt=system_prompt,
        generation=GenerationSettings(max_new_tokens=max_new_tokens, enable_thinking=enable_thinking, stream_output=stream_output),
    )
    sandbox = get_sandbox_service(config)
    model_loaded = sandbox.is_model_loaded()

    # --- Handle Load Model button ---
    if load_clicked:
        with st.spinner(f"Loading model {model_id}... This may download weights and take several minutes."):
            try:
                sandbox.load_model(model_id)
                st.session_state["_loaded_model_id"] = model_id
                st.session_state["_model_ready"] = True
                _model_ready = True
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to load model: {exc}")
                st.session_state["_model_ready"] = False
                _model_ready = False

    left, right = st.columns([1.2, 0.8])

    with left:
        st.markdown('<div class="support-chip">Native</div>', unsafe_allow_html=True)

        active_conversation_key = _conversation_key(
            model_id=model_id,
            system_prompt=system_prompt,
        )
        model_history, ui_history = _get_history_for_key(active_conversation_key)
        conversation_history_slot = st.empty()

        left_controls = st.columns([0.7, 0.3])
        with left_controls[0]:
            turn_counter_slot = st.empty()
            turn_counter_slot.caption(f"Conversation turns: {len(ui_history) // 2}")
        with left_controls[1]:
            if st.button("Clear conversation", use_container_width=True):
                _clear_history_for_key(active_conversation_key)
                model_history, ui_history = _get_history_for_key(active_conversation_key)
        with conversation_history_slot.container():
            _render_conversation_history(ui_history)

        pending_exchange_slot = st.empty()

        _generating = st.session_state.get("_generating", False)
        _input_key = st.session_state.get("_uploader_key", 0)

        # ---- Media attachment tabs ----
        _upload_types_image = ["png", "jpg", "jpeg", "webp"]
        _upload_types_audio = ["wav", "mp3", "flac", "ogg", "m4a"]
        _upload_types_video = ["mp4", "mov", "avi", "mkv", "webm"]
        _allowed_upload_types = [
            *_upload_types_image,
            *(_upload_types_audio if caps.audio else []),
            *(_upload_types_video if caps.video else []),
        ]

        uploaded_file = None
        image_url_input: str = ""
        pasted_image_data = None

        _media_tab_labels = ["📁 Upload", "🔗 URL", "📋 Paste"]
        _tab_upload, _tab_url, _tab_paste = st.tabs(_media_tab_labels)

        with _tab_upload:
            _upload_help_parts = ["image"]
            if caps.audio:
                _upload_help_parts.append("audio")
            if caps.video:
                _upload_help_parts.append("video (auto-sampled into frames)")
            uploaded_file = st.file_uploader(
                "Attach media (optional)",
                type=_allowed_upload_types,
                help=f"Supported for this model: {', '.join(_upload_help_parts)}.",
                disabled=_generating,
                key=f"upload_{_input_key}",
            )

        with _tab_url:
            if caps.image:
                image_url_input = st.text_input(
                    "Image URL",
                    placeholder="https://example.com/image.jpg",
                    help="Paste a public https:// image URL. The model processor fetches it directly.",
                    disabled=_generating,
                    key=f"url_input_{_input_key}",
                )
            else:
                st.caption("This model does not support image input.")

        with _tab_paste:
            if _PASTE_AVAILABLE:
                _paste_result = _paste_image_button("📋 Paste image from clipboard")
                if _paste_result is not None and getattr(_paste_result, "image_data", None) is not None:
                    pasted_image_data = _paste_result.image_data
                    st.image(pasted_image_data, caption="Pasted image", width=200)
            else:
                st.info(
                    "Clipboard paste requires the `streamlit-paste-button` package. "
                    "Install it with: `pip install streamlit-paste-button`",
                    icon="ℹ️",
                )

        user_prompt = st.chat_input(
            "Generating response, please wait..." if _generating else "Type your message and press Enter to run.",
            disabled=not _model_ready or _generating,
        ) or ""

        # --- Phase 1: on fresh submission, capture everything and trigger generation rerun ---
        if user_prompt.strip() and _model_ready and not _generating:
            _pending: dict = {
                "prompt": user_prompt,
                "uploaded_path": None,
                "uploaded_name": None,
                "image_urls": list(filter(None, [image_url_input.strip()])),
                "pasted_image_path": None,
            }
            if uploaded_file is not None:
                _pending["uploaded_path"] = str(persist_upload(uploaded_file))
                _pending["uploaded_name"] = uploaded_file.name
            if pasted_image_data is not None:
                from PIL import Image as _PIL_Image
                _paste_tmp = Path(tempfile.mktemp(suffix=".png"))
                _img = pasted_image_data if isinstance(pasted_image_data, _PIL_Image.Image) else _PIL_Image.fromarray(pasted_image_data)
                _img.save(_paste_tmp)
                _pending["pasted_image_path"] = str(_paste_tmp)
            st.session_state["_pending_turn"] = _pending
            st.session_state["_generating"] = True
            st.session_state["_uploader_key"] = _input_key + 1
            st.rerun()

        # --- Phase 2: generation rerun — run_clicked uses stored pending ---
        run_clicked = _generating and bool(st.session_state.get("_pending_turn", {}).get("prompt", "").strip())

        if not _model_ready:
            st.caption("Load a model from the sidebar before running.")


    with right:
        st.subheader("Model Capabilities")
        st.markdown(f"**{selected_model_label}**")
        for _cap_label, _supported in [("🖼️ Images", caps.image), ("🔊 Audio", caps.audio), ("🎬 Video", caps.video)]:
            st.markdown(f"{'✅' if _supported else '❌'} {_cap_label}")
        if _model_ready:
            st.success(f"Runtime status: {model_id} loaded and ready.")
        elif model_loaded:
            st.warning(f"Runtime status: server is up but {model_id} is not pre-loaded. Click Load Model in the sidebar.")
        else:
            st.warning(
                f"Runtime status: cold start for {model_id}. Click Load Model in the sidebar to prepare the model before running."
            )
        st.caption(
            "The app now reports progress stages in both the server logs and the UI: runtime check, processor load, model load, input prep, generation, and decoding."
        )

    if not run_clicked:
        return

    # Phase 2: all input data comes from session state — inputs are already cleared
    pending = st.session_state["_pending_turn"]
    user_prompt = pending["prompt"]
    attachment_labels: list[str] = []
    attachment = TurnAttachment()
    frame_paths: list[Path] = []
    _pending_uploaded_path = Path(pending["uploaded_path"]) if pending.get("uploaded_path") else None
    _pending_uploaded_name: str | None = pending.get("uploaded_name")

    run_status = st.status("Queued sandbox run.", expanded=True)
    run_progress = st.progress(0, text="Waiting to start...")

    def emit_progress(stage: str, progress_value: float, message: str) -> None:
        LOGGER.info("ui-progress [%s] %s", stage, message)
        run_status.write(message)
        run_progress.progress(min(max(progress_value, 0.0), 1.0), text=message)

    emit_progress("start", 0.02, "Preparing request...")
    if _pending_uploaded_path is not None and _pending_uploaded_name is not None:
        ext = Path(_pending_uploaded_name).suffix.lower()
        label = _pending_uploaded_name
        if ext in _IMAGE_EXTS:
            attachment.image_paths.append(_pending_uploaded_path)
            attachment_labels.append(f"image: {label}")
        elif ext in _AUDIO_EXTS:
            attachment.audio_path = _pending_uploaded_path
            attachment_labels.append(f"audio: {label}")
        elif ext in _VIDEO_EXTS:
            emit_progress("video", 0.10, "Extracting representative video frames...")
            frame_paths = extract_video_frames(_pending_uploaded_path)
            attachment.video_frame_paths = frame_paths
            attachment_labels.append(f"video: {label} ({len(frame_paths)} frames)")

    for _url in pending.get("image_urls", []):
        attachment.image_urls.append(_url)
        attachment_labels.append(f"image URL: {_url}")

    if pending.get("pasted_image_path"):
        attachment.image_paths.append(Path(pending["pasted_image_path"]))
        attachment_labels.append("image: pasted from clipboard")


    pending_exchange_slot.empty()
    with pending_exchange_slot.container():
        live_response_placeholder = _render_pending_exchange(
            user_prompt=user_prompt,
            attachment_labels=attachment_labels,
        )
    if stream_output:
        live_response_placeholder.caption("No tokens received yet.")
    else:
        live_response_placeholder.caption("Streaming disabled. The full response will appear when generation completes.")

    def emit_partial_text(text: str) -> None:
        preview = text.strip() or "Receiving generated tokens..."
        live_response_placeholder.markdown(f"**Live response**\n\n{preview}")

    started_at = perf_counter()

    try:
        result = sandbox.run(
            user_prompt=user_prompt,
            attachment=attachment,
            prior_turns=model_history,
            progress_callback=emit_progress,
            token_callback=emit_partial_text if stream_output else None,
        )
        elapsed_seconds = perf_counter() - started_at
        run_status.update(label="Sandbox run complete.", state="complete", expanded=False)
        live_response_placeholder.empty()

        # Append the current turn to both histories.
        # model_history receives full content parts (including media) for the next request.
        # ui_history receives display-only data for rendering the thread.
        user_content: list[dict] = []
        for path in attachment.image_paths:
            user_content.append({"type": "image", "url": path.as_posix()})
        for url in attachment.image_urls:
            user_content.append({"type": "image", "url": url})
        if attachment.audio_path is not None:
            user_content.append({"type": "audio", "audio": attachment.audio_path.as_posix()})
        for path in attachment.video_frame_paths:
            user_content.append({"type": "image", "url": path.as_posix()})
        user_content.append({"type": "text", "text": result.prompt_used})
        model_history.append({"role": "user", "content": user_content})
        model_history.append({"role": "assistant", "content": [{"type": "text", "text": result.response_text}]})
        ui_history.append({"role": "user", "text": user_prompt, "attachment_labels": attachment_labels})
        ui_history.append({"role": "assistant", "text": result.response_text})
        st.session_state["_generating"] = False
        st.session_state.pop("_pending_turn", None)
        st.rerun()

        st.subheader("Result")
        st.caption(f"Support level: {result.support_level}")
        st.caption(f"Response time: {elapsed_seconds:.2f} seconds")
        metadata = result.run_metadata
        timings = metadata.get("timings", {}) if isinstance(metadata.get("timings"), dict) else {}
        memory = metadata.get("memory", {}) if isinstance(metadata.get("memory"), dict) else {}
        prompt_char_count = metadata.get("prompt_char_count")
        response_char_count = metadata.get("response_char_count")
        generation_tokens_per_second = metadata.get("output_tokens_per_second")
        tokens_per_second = _calculate_tokens_per_second(result.output_token_count, elapsed_seconds)
        token_parts: list[str] = []
        if result.input_token_count is not None:
            token_parts.append(f"Input tokens: {result.input_token_count}")
        if result.output_token_count is not None:
            token_parts.append(f"Output tokens: {result.output_token_count}")
        if result.total_token_count is not None:
            token_parts.append(f"Total tokens: {result.total_token_count}")
        if tokens_per_second is not None:
            token_parts.append(f"Output tok/s: {tokens_per_second:.2f}")
        if token_parts:
            st.caption(" | ".join(token_parts))
        char_parts: list[str] = []
        if isinstance(prompt_char_count, int):
            char_parts.append(f"Prompt chars: {prompt_char_count}")
        if isinstance(response_char_count, int):
            char_parts.append(f"Response chars: {response_char_count}")
        if char_parts:
            st.caption(" | ".join(char_parts))
        timing_parts: list[str] = []
        runtime_load_seconds = timings.get("runtime_load_seconds")
        generation_seconds = timings.get("generation_seconds")
        decode_seconds = timings.get("decode_seconds")
        if isinstance(runtime_load_seconds, (int, float)):
            timing_parts.append(f"Load: {runtime_load_seconds:.2f}s")
        if isinstance(generation_seconds, (int, float)):
            timing_parts.append(f"Generate: {generation_seconds:.2f}s")
        if isinstance(decode_seconds, (int, float)):
            timing_parts.append(f"Decode: {decode_seconds:.2f}s")
        if isinstance(generation_tokens_per_second, (int, float)):
            timing_parts.append(f"Gen tok/s: {generation_tokens_per_second:.2f}")
        if timing_parts:
            st.caption(" | ".join(timing_parts))
        memory_parts: list[str] = []
        process_rss_delta_mb = memory.get("process_rss_delta_mb")
        if isinstance(process_rss_delta_mb, (int, float)):
            memory_parts.append(f"RAM delta: {process_rss_delta_mb:.2f} MB")
        cuda_peak_allocated_mb = memory.get("cuda_peak_allocated_mb")
        if isinstance(cuda_peak_allocated_mb, (int, float)):
            memory_parts.append(f"Peak VRAM: {cuda_peak_allocated_mb:.2f} MB")
        if memory_parts:
            st.caption(" | ".join(memory_parts))
        st.caption("The full response is shown inline next to the prompt or in the conversation thread above.")

        with st.expander("Run metadata"):
            st.json(
                {
                    "model_id": result.model_id,
                    "runtime_state": "cold-start" if result.was_cold_start else "warm-start",
                    "response_time_seconds": round(elapsed_seconds, 3),
                    "output_tokens_per_second": round(tokens_per_second, 3) if tokens_per_second is not None else None,
                    "input_token_count": result.input_token_count,
                    "output_token_count": result.output_token_count,
                    "total_token_count": result.total_token_count,
                    "prompt_char_count": prompt_char_count,
                    "response_char_count": response_char_count,
                    "runtime_load_seconds": timings.get("runtime_load_seconds"),
                    "prepare_seconds": timings.get("prepare_seconds"),
                    "generation_seconds": timings.get("generation_seconds"),
                    "decode_seconds": timings.get("decode_seconds"),
                    "service_seconds": timings.get("service_seconds"),
                    "generation_output_tokens_per_second": generation_tokens_per_second,
                    "memory": memory,
                    "support_level": result.support_level,
                    "system_prompt": system_prompt,
                    "conversation_turn_count": len(ui_history) // 2,
                    "enable_thinking": enable_thinking,
                    "stream_output": stream_output,
                    "max_new_tokens": max_new_tokens,
                    "temperature": config.generation.temperature,
                    "top_p": config.generation.top_p,
                    "top_k": config.generation.top_k,
                    "uploaded_file": _pending_uploaded_name,
                    "sampled_frame_count": len(frame_paths),
                }
            )

        with st.expander("Prompt used"):
            st.code(result.prompt_used)

        if frame_paths:
            with st.expander("Sampled frames"):
                st.write(f"Extracted {len(frame_paths)} frames for analysis.")
                st.image([str(path) for path in frame_paths], use_container_width=True)
    except Exception as error:
        run_status.update(label="Sandbox run failed.", state="error", expanded=True)
        live_response_placeholder.empty()
        st.error(str(error))
        st.session_state["_generating"] = False
        st.session_state.pop("_pending_turn", None)
        st.rerun()


if __name__ == "__main__":
    main()