from __future__ import annotations

import logging
from pathlib import Path
import sys
import tempfile
from time import perf_counter

from env_bootstrap import bootstrap_environment

bootstrap_environment()

import streamlit as st
from streamlit.components.v1 import html as _components_html

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_sandbox.config import AppConfig, DEFAULT_MODEL_ID, DEFAULT_SYSTEM_PROMPT, GenerationSettings, SERVING_URL
from ai_sandbox.domain import RunResult
from ai_sandbox.media import extract_video_frames, persist_upload
from ai_sandbox.model_profiles import MODEL_LABELS, get_capabilities, get_label_for_model_id, get_model_id, model_labels_for_backend
from ai_sandbox.services.sandbox_service import SandboxService, TurnAttachment
from ai_sandbox.services.serving_client import _ensure_data_uri_or_url
from ai_sandbox.services.serving_client import ServingClient

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
    page_title="AI Workbench",
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
        /* Chat window: scrollable history area */
        [data-testid="stVerticalBlockBorderWrapper"] {
            border: 1px solid rgba(60, 80, 110, 0.10) !important;
            border-radius: 16px !important;
            background: rgba(255, 255, 255, 0.50) !important;
            padding: 0.25rem 0.25rem !important;
        }
        /* Media attachment tabs — compact */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.25rem;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 0.25rem 0.65rem;
            font-size: 0.82rem;
        }
        /* Chat input area: subtle top separator */
        .chat-input-area {
            margin-top: 0.4rem;
            padding-top: 0.35rem;
            border-top: 1px solid rgba(60, 80, 110, 0.10);
        }
        /* Code blocks: horizontal scroll by default */
        pre {
            overflow-x: auto !important;
            max-width: 100% !important;
        }
        /* When wrap mode is active, switch to word-wrapping */
        .wrap-code-blocks pre {
            white-space: pre-wrap !important;
            word-wrap: break-word !important;
            overflow-x: visible !important;
        }
        .wrap-code-blocks pre code {
            white-space: pre-wrap !important;
            word-wrap: break-word !important;
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


def _render_run_history(ui_history: list[dict]) -> None:
    """Render per-turn run summaries in the right panel.

    Each turn shows as a collapsed expander except the last one, which is
    expanded so the user can see the most-recent result immediately.
    """
    turns = [
        (i, msg)
        for i, msg in enumerate(ui_history)
        if msg.get("role") == "assistant" and msg.get("run_summary")
    ]
    if not turns:
        st.caption("Run metrics will appear here after the first response.")
        return

    for list_idx, (history_idx, msg) in enumerate(turns):
        s = msg["run_summary"]
        is_last = list_idx == len(turns) - 1
        turn_num = list_idx + 1
        elapsed = s.get("response_time_seconds")
        tps = s.get("output_tokens_per_second")
        # Brief label: Turn N · Xs · N tok/s
        label_parts = [f"Turn {turn_num}"]
        if elapsed is not None:
            label_parts.append(f"{elapsed:.2f}s")
        if tps is not None:
            label_parts.append(f"{tps:.1f} tok/s")
        label = " · ".join(label_parts)

        with st.expander(label, expanded=is_last):
            # Token counts
            token_parts: list[str] = []
            if s.get("input_token_count") is not None:
                token_parts.append(f"In: **{s['input_token_count']}**")
            if s.get("output_token_count") is not None:
                token_parts.append(f"Out: **{s['output_token_count']}**")
            if s.get("total_token_count") is not None:
                token_parts.append(f"Total: **{s['total_token_count']}**")
            if token_parts:
                st.markdown("  ".join(token_parts))
            # Timing row
            timing_parts: list[str] = []
            timings = s.get("timings", {})
            if isinstance(timings.get("runtime_load_seconds"), (int, float)):
                timing_parts.append(f"Load: {timings['runtime_load_seconds']:.2f}s")
            if isinstance(timings.get("generation_seconds"), (int, float)):
                timing_parts.append(f"Gen: {timings['generation_seconds']:.2f}s")
            if isinstance(timings.get("decode_seconds"), (int, float)):
                timing_parts.append(f"Decode: {timings['decode_seconds']:.2f}s")
            if timing_parts:
                st.caption(" | ".join(timing_parts))
            # Memory
            memory = s.get("memory", {})
            mem_parts: list[str] = []
            if isinstance(memory.get("process_rss_delta_mb"), (int, float)):
                mem_parts.append(f"RAM Δ: {memory['process_rss_delta_mb']:.1f} MB")
            if isinstance(memory.get("cuda_peak_allocated_mb"), (int, float)):
                mem_parts.append(f"Peak VRAM: {memory['cuda_peak_allocated_mb']:.1f} MB")
            if mem_parts:
                st.caption(" | ".join(mem_parts))
            # Misc flags
            misc: list[str] = []
            if s.get("runtime_state") == "cold-start":
                misc.append("❄️ cold-start")
            if s.get("support_level"):
                misc.append(f"support: {s['support_level']}")
            if s.get("uploaded_file"):
                misc.append(f"file: {s['uploaded_file']}")
            if misc:
                st.caption("  ·  ".join(misc))
            # Raw JSON in nested expander
            with st.expander("Full metadata", expanded=False):
                st.json(s)


def main() -> None:
    inject_styles()

    # --- Early backend probe (before sidebar) ---
    # Detect which model the server is actually serving so the dropdown
    # auto-selects it.  vLLM loads ONE model at startup; the dropdown is
    # informational only.  Windows backend can switch models on demand.
    _probe = ServingClient(base_url=SERVING_URL)
    _server_healthy = _probe.is_healthy()
    _backend_mode = _probe.detect_backend_mode() if _server_healthy else "vllm"
    _active_model: str | None = _probe.get_active_model_id() if _server_healthy else None
    _active_label: str | None = get_label_for_model_id(_active_model) if _active_model else None
    _is_native_backend = _backend_mode == "native"
    _dropdown_labels = model_labels_for_backend(_backend_mode)

    # Choose the dropdown default: server's model if known, else first entry.
    if _active_label and _active_label in _dropdown_labels:
        _default_index = _dropdown_labels.index(_active_label)
    else:
        _default_index = 0

    with st.sidebar:
        st.header("Settings")
        selected_model_label = st.selectbox(
            "Model",
            options=_dropdown_labels,
            index=_default_index,
            help="Native backend: switches model on demand. vLLM: fixed at server startup." if _server_healthy else None,
        )
        model_id = get_model_id(selected_model_label, fallback=DEFAULT_MODEL_ID)

        # --- Windows backend: allow model switching ---
        if _is_native_backend and _active_model and model_id != _active_model:
            if st.button("🔄 Load selected model", use_container_width=True):
                with st.spinner(f"Loading {selected_model_label}..."):
                    try:
                        _probe.load_model(model_id)
                        st.success(f"Loaded {selected_model_label}")
                        # Refresh probe state after load
                        _active_model = _probe.get_active_model_id()
                        _active_label = get_label_for_model_id(_active_model) if _active_model else None
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to load model: {e}")

        caps = get_capabilities(selected_model_label)

        _sidebar_status_slot = st.empty()

        if caps.vram_gb_bf16 > 0:
            st.caption(f"~{caps.vram_gb_bf16:.0f} GB VRAM at BF16 (managed by backend).")

        if "_system_prompt_input" not in st.session_state:
            st.session_state["_system_prompt_input"] = DEFAULT_SYSTEM_PROMPT
        system_prompt = st.text_area(
            "System prompt",
            height=160,
            key="_system_prompt_input",
        )
        _apply_col_spacer, _apply_col_btn = st.columns([0.7, 0.3])
        with _apply_col_btn:
            st.button("Apply", key="_apply_system_prompt", help="Apply system prompt changes (or press Ctrl+Enter)", type="secondary", use_container_width=True)
        max_new_tokens = st.slider("Max new tokens", min_value=64, max_value=2048, value=256, step=64,
                                     help="Upper limit on generated tokens. The model may stop earlier if it decides the answer is complete (EOS token).")
        with st.expander("Sampling parameters", expanded=False):
            temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=1.0, step=0.05,
                                    help="Controls randomness. 0 = greedy/deterministic, 1 = maximum randomness.")
            top_p = st.slider("Top-p (nucleus)", min_value=0.0, max_value=1.0, value=0.95, step=0.01,
                              help="Cumulative probability cutoff. Lower = only highest-probability tokens considered.")
            top_k = st.slider("Top-k", min_value=1, max_value=200, value=64, step=1,
                              help="Number of highest-probability tokens to consider at each step.")
        enable_thinking = st.toggle("Enable thinking", value=False)
        stream_output = st.checkbox("Stream text response to UI", value=True)
        wrap_code = st.checkbox("Wrap code blocks", value=False, help="Toggle word-wrap on code blocks. Off = horizontal scroll.")
        st.caption(f"Active model: {_active_model or model_id}")
        st.caption(f"Backend: {'Native (Transformers)' if _is_native_backend else 'vLLM'} · {SERVING_URL}")

    # Apply wrap-code-blocks CSS class toggle via JS
    _wrap_action = "add" if wrap_code else "remove"
    _components_html(
        f"""<script>
        window.parent.document.querySelector('.stApp').classList.{_wrap_action}('wrap-code-blocks');
        </script>""",
        height=0,
    )

    # Use the effective model: on Windows backend the user picks freely;
    # on vLLM the server's model overrides the dropdown.
    if _is_native_backend:
        _effective_model_id = model_id
        _effective_label = selected_model_label
    else:
        _effective_model_id = _active_model or model_id
        _effective_label = (_active_label or selected_model_label) if _active_label else selected_model_label
    _effective_caps = get_capabilities(_effective_label)
    _family = _model_family_label(_effective_label)
    _media_note = "Attach images" + (", audio," if _effective_caps.audio else "") + (" video," if _effective_caps.video else "") + " or other media to any message. Responses are always text."
    st.markdown('<div class="hero">', unsafe_allow_html=True)
    st.markdown(f'<div class="mono-label">{_family} Sandbox</div>', unsafe_allow_html=True)
    st.title("AI Workbench")
    st.write(_media_note)
    st.markdown("</div>", unsafe_allow_html=True)

    config = AppConfig(
        model_id=_effective_model_id,
        system_prompt=system_prompt,
        generation=GenerationSettings(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            enable_thinking=enable_thinking,
            stream_output=stream_output,
        ),
    )
    sandbox = get_sandbox_service(config)

    # --- Backend status (uses the early probe results) ---
    _model_ready = _server_healthy
    _model_mismatch = bool(not _is_native_backend and _active_model and _active_model != model_id)

    # Fill the sidebar status placeholder created earlier.
    with _sidebar_status_slot.container():
        if _model_ready and _is_native_backend:
            st.success(f"Connected — {_active_model or model_id} · switch freely", icon="\u2705")
        elif _model_ready and not _model_mismatch:
            st.success(f"Connected — {_active_model or model_id}", icon="\u2705")
        elif _model_ready and _model_mismatch:
            st.info(
                f"Server has **{_active_label or _active_model}** loaded. "
                f"Select it in the dropdown, or restart the backend to switch models.",
                icon="\u2139\uFE0F",
            )
        else:
            st.error("Backend offline", icon="\u274C")

    left, right = st.columns([1.2, 0.8])

    with left:
        active_conversation_key = _conversation_key(
            model_id=model_id,
            system_prompt=system_prompt,
        )
        model_history, ui_history = _get_history_for_key(active_conversation_key)

        # ---- Chat header: chip + turn counter + clear button ----
        _hdr_chip, _hdr_turns, _hdr_clear = st.columns([0.32, 0.42, 0.26])
        with _hdr_chip:
            st.markdown('<div class="support-chip">Native</div>', unsafe_allow_html=True)
        _turn_counter_slot = _hdr_turns.empty()
        _turn_counter_slot.caption(f"Conversation turns: {len(ui_history) // 2}")
        with _hdr_clear:
            if st.button("Clear", use_container_width=True, key="clear_conv_btn"):
                _clear_history_for_key(active_conversation_key)
                model_history, ui_history = _get_history_for_key(active_conversation_key)
                st.rerun()

        # ---- Scrollable chat history window ----
        _chat_window = st.container(height=500, border=False)
        with _chat_window:
            _render_conversation_history(ui_history)
            pending_exchange_slot = st.empty()

        # Auto-scroll to the latest message
        _components_html(
            """<script>
(function() {
    var candidates = window.parent.document.querySelectorAll('[data-testid="stVerticalBlockBorderWrapper"]');
    if (candidates.length > 0) {
        var el = candidates[0];
        el.scrollTop = el.scrollHeight;
    }
})();
</script>""",
            height=0,
        )

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
            st.caption(f"Waiting for backend at {SERVING_URL} — start the server to begin.")


    with right:
        # ---- Model Capabilities (compact) ----
        with st.expander("Model Capabilities", expanded=True):
            _display_label = (_active_label or selected_model_label) if _model_mismatch else selected_model_label
            _display_caps = get_capabilities(_display_label)
            st.markdown(f"**{_display_label}**")
            for _cap_label, _supported in [("🖼️ Images", _display_caps.image), ("🔊 Audio", _display_caps.audio), ("🎬 Video", _display_caps.video)]:
                st.markdown(f"{'✅' if _supported else '❌'} {_cap_label}")
            if _model_ready:
                st.success(f"**{_active_model or model_id}** ready", icon="✅")
            elif _server_healthy:
                st.info("Server up, waiting for model info.")
            else:
                st.warning(
                    f"Cannot reach backend at {SERVING_URL}. "
                    f"Start with `vllm-serving/start_vllm.ps1` (WSL2) or `start_server.ps1` (Windows)."
                )

        # ---- Per-turn run metrics ----
        st.markdown("**Run History**")
        _render_run_history(ui_history)

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
    LOGGER.info(
        "TURN_DIAG: pending uploaded_path=%r, uploaded_name=%r, image_urls=%r, pasted=%r",
        pending.get("uploaded_path"), pending.get("uploaded_name"),
        pending.get("image_urls"), pending.get("pasted_image_path"),
    )
    if _pending_uploaded_path is not None and _pending_uploaded_name is not None:
        ext = Path(_pending_uploaded_name).suffix.lower()
        label = _pending_uploaded_name
        LOGGER.info(
            "TURN_DIAG: uploaded file=%s ext=%s exists=%s size=%s",
            _pending_uploaded_path, ext,
            _pending_uploaded_path.exists(),
            _pending_uploaded_path.stat().st_size if _pending_uploaded_path.exists() else 'N/A',
        )
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

    LOGGER.info(
        "TURN_DIAG: attachment image_paths=%r, image_urls=%r, prior_turns=%d",
        [str(p) for p in attachment.image_paths],
        attachment.image_urls,
        len(model_history),
    )
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
        # model_history stores data URIs (not file paths) so prior-turn images
        # survive across Streamlit reruns and temp-file cleanup.
        # ui_history receives display-only data for rendering the thread.
        user_content: list[dict] = []
        for path in attachment.image_paths:
            user_content.append({"type": "image", "url": _ensure_data_uri_or_url(path.as_posix())})
        for url in attachment.image_urls:
            user_content.append({"type": "image", "url": _ensure_data_uri_or_url(url)})
        if attachment.audio_path is not None:
            user_content.append({"type": "audio", "audio": attachment.audio_path.as_posix()})
        for path in attachment.video_frame_paths:
            user_content.append({"type": "image", "url": _ensure_data_uri_or_url(path.as_posix())})
        user_content.append({"type": "text", "text": result.prompt_used})
        # Build the run summary to persist in session state before the rerun.
        metadata = result.run_metadata
        timings = metadata.get("timings", {}) if isinstance(metadata.get("timings"), dict) else {}
        memory = metadata.get("memory", {}) if isinstance(metadata.get("memory"), dict) else {}
        tokens_per_second = _calculate_tokens_per_second(result.output_token_count, elapsed_seconds)
        _run_summary = {
            "model_id": result.model_id,
            "runtime_state": "cold-start" if result.was_cold_start else "warm-start",
            "response_time_seconds": round(elapsed_seconds, 3),
            "output_tokens_per_second": round(tokens_per_second, 3) if tokens_per_second is not None else None,
            "input_token_count": result.input_token_count,
            "output_token_count": result.output_token_count,
            "total_token_count": result.total_token_count,
            "prompt_char_count": metadata.get("prompt_char_count"),
            "response_char_count": metadata.get("response_char_count"),
            "timings": timings,
            "generation_output_tokens_per_second": metadata.get("output_tokens_per_second"),
            "memory": memory,
            "support_level": result.support_level,
            "system_prompt": system_prompt,
            "conversation_turn_count": len(ui_history) // 2 + 1,
            "enable_thinking": enable_thinking,
            "stream_output": stream_output,
            "max_new_tokens": max_new_tokens,
            "temperature": config.generation.temperature,
            "top_p": config.generation.top_p,
            "top_k": config.generation.top_k,
            "uploaded_file": _pending_uploaded_name,
            "sampled_frame_count": len(frame_paths),
            "prompt_used": result.prompt_used,
        }

        model_history.append({"role": "user", "content": user_content})
        model_history.append({"role": "assistant", "content": [{"type": "text", "text": result.response_text}]})
        ui_history.append({"role": "user", "text": user_prompt, "attachment_labels": attachment_labels})
        ui_history.append({"role": "assistant", "text": result.response_text, "run_summary": _run_summary})
        st.session_state["_generating"] = False
        st.session_state.pop("_pending_turn", None)
        st.rerun()
    except Exception as error:
        run_status.update(label="Sandbox run failed.", state="error", expanded=True)
        live_response_placeholder.empty()
        st.error(str(error))
        st.session_state["_generating"] = False
        st.session_state.pop("_pending_turn", None)
        st.rerun()


if __name__ == "__main__":
    main()