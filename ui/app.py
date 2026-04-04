from __future__ import annotations

import logging
from pathlib import Path
import sys
from time import perf_counter

from env_bootstrap import bootstrap_environment

bootstrap_environment()

import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gemma_sandbox.config import AppConfig, DEFAULT_MODEL_ID, DEFAULT_SYSTEM_PROMPT, GenerationSettings, MODEL_OPTIONS, SERVING_URL
from gemma_sandbox.domain import Ability
from gemma_sandbox.media import extract_video_frames, persist_upload
from gemma_sandbox.prompts import PERSONA_PRESETS
from gemma_sandbox.services.sandbox_service import SandboxService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

LOGGER = logging.getLogger(__name__)


st.set_page_config(
    page_title="Gemma Sandbox Arena",
    page_icon="AI",
    layout="wide",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;600&display=swap');
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


def _format_optional_float(value: object, suffix: str = "") -> str:
    if isinstance(value, (int, float)):
        return f"{value:.2f}{suffix}"
    return "unavailable"


CONVERSATION_CAPABLE_ABILITIES = {
    Ability.TEXT_TO_TEXT,
    Ability.TEXT_TO_IMAGE,
    Ability.TEXT_TO_VIDEO,
    Ability.TEXT_TO_AUDIO,
}


def _conversation_key(
    *,
    ability: Ability,
    model_id: str,
    system_prompt: str,
) -> str:
    return "|".join((ability.value, model_id, system_prompt.strip()))


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


def _render_conversation_history(history: list[dict[str, str]]) -> None:
    if not history:
        st.caption("Conversation is empty. Ask the first question to start a contextual thread.")
        return
    for message in history:
        role = message.get("role", "assistant")
        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(message.get("content", ""))


def _render_pending_exchange(
    *,
    conversation_mode: bool,
    user_prompt: str,
) -> tuple[st.delta_generator.DeltaGenerator, st.delta_generator.DeltaGenerator | None]:
    if conversation_mode:
        with st.chat_message("user"):
            st.markdown(user_prompt)
        with st.chat_message("assistant"):
            placeholder = st.empty()
            return placeholder, None

    wrapper = st.container()
    with wrapper:
        st.markdown("**Latest exchange**")
        with st.chat_message("user"):
            st.markdown(user_prompt)
        with st.chat_message("assistant"):
            placeholder = st.empty()
    return placeholder, wrapper


def _render_latest_exchange(user_prompt: str, response_text: str) -> None:
    st.markdown("**Latest exchange**")
    with st.chat_message("user"):
        st.markdown(user_prompt)
    with st.chat_message("assistant"):
        st.markdown(response_text)


def main() -> None:
    inject_styles()
    st.markdown('<div class="hero">', unsafe_allow_html=True)
    st.markdown('<div class="mono-label">Gemma 4 Sandbox</div>', unsafe_allow_html=True)
    st.title("Gemma Sandbox Arena")
    st.write(
        "A game-like control room for testing Gemma 4 across native multimodal understanding workflows "
        "and clearly labeled simulated media-planning workflows."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    with st.sidebar:
        st.header("Mission Settings")
        model_labels = [label for label, _model_id in MODEL_OPTIONS]
        model_id_to_label = {model_id: label for label, model_id in MODEL_OPTIONS}
        selected_model_label = st.selectbox(
            "Gemma model",
            options=[*model_labels, "Custom model ID"],
            index=(model_labels.index(model_id_to_label[DEFAULT_MODEL_ID]) if DEFAULT_MODEL_ID in model_id_to_label else len(model_labels)),
        )
        if selected_model_label == "Custom model ID":
            model_id = st.text_input("Custom model ID", value=DEFAULT_MODEL_ID)
        else:
            model_id = dict(MODEL_OPTIONS)[selected_model_label]

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

        ability = Ability(
            st.selectbox(
                "Choose ability",
                options=[ability.value for ability in Ability],
                format_func=lambda value: value.replace("-", " ").title(),
            )
        )
        conversation_supported = ability in CONVERSATION_CAPABLE_ABILITIES
        conversation_mode = st.checkbox(
            "Conversation mode",
            value=False,
            disabled=not conversation_supported,
            help="Keep prior text turns in context for follow-up questions. Currently available for text and simulated text modes.",
        )
        preset_name = st.selectbox("Assistant persona", options=list(PERSONA_PRESETS))
        preset_text = PERSONA_PRESETS[preset_name]
        if "_last_persona" not in st.session_state:
            st.session_state["_last_persona"] = preset_name
        if "_system_prompt_input" not in st.session_state:
            st.session_state["_system_prompt_input"] = preset_text or DEFAULT_SYSTEM_PROMPT
        if preset_name != st.session_state["_last_persona"]:
            st.session_state["_last_persona"] = preset_name
            st.session_state["_system_prompt_input"] = preset_text
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
        spec = sandbox.get_ability_spec(ability)
        if not conversation_supported:
            st.caption("Conversation mode is currently limited to text and simulated text abilities. Upload-based modes still run as isolated requests.")
        st.markdown(f'<div class="support-chip">{spec.support_level}</div>', unsafe_allow_html=True)
        st.write(spec.summary)

        active_conversation_key = _conversation_key(
            ability=ability,
            model_id=model_id,
            system_prompt=system_prompt,
        )
        model_history, ui_history = _get_history_for_key(active_conversation_key)
        conversation_history_slot = st.empty()
        pending_exchange_slot = st.empty()
        if conversation_mode:
            left_controls = st.columns([0.7, 0.3])
            with left_controls[0]:
                st.caption(f"Conversation turns: {len(ui_history) // 2}")
            with left_controls[1]:
                if st.button("Clear conversation", use_container_width=True):
                    _clear_history_for_key(active_conversation_key)
                    model_history, ui_history = _get_history_for_key(active_conversation_key)
            with conversation_history_slot.container():
                _render_conversation_history(ui_history)

        latest_exchange_slot = st.empty()

        user_prompt = ""
        run_clicked = False
        if conversation_mode:
            user_prompt = st.chat_input(
                "Ask a follow-up question without losing context",
                disabled=not _model_ready,
            ) or ""
            run_clicked = bool(user_prompt.strip())
        else:
            user_prompt = st.text_area(
                "Prompt",
                height=180,
                placeholder="Describe the task you want the sandbox to run.",
            )

        uploaded_file = None
        if ability is Ability.IMAGE_TO_TEXT:
            uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "webp"])
        elif ability is Ability.AUDIO_TO_TEXT:
            uploaded_file = st.file_uploader("Upload audio", type=["wav", "mp3", "flac", "ogg", "m4a"])
        elif ability is Ability.VIDEO_TO_TEXT:
            uploaded_file = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv", "webm"])

        if not conversation_mode:
            run_clicked = st.button(
                "Run Sandbox",
                type="primary",
                use_container_width=True,
                disabled=not _model_ready,
            )
            if not _model_ready:
                st.caption("Load a model from the sidebar before running.")

    with right:
        st.subheader("Mode Guide")
        st.write(
            "Native modes call Gemma directly. Experimental mode samples video frames. Simulated modes use Gemma to plan artifacts for a future media generator."
        )
        st.info(
            "Recommended simulator usage: Multimodal Situation Room. It turns the app into an operator console for analysis, transcription, prompt design, and storyboard planning."
        )
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

    uploaded_path: Path | None = None
    frame_paths: list[Path] | None = None
    run_status = st.status("Queued sandbox run.", expanded=True)
    run_progress = st.progress(0, text="Waiting to start...")
    if conversation_mode:
        pending_exchange_slot.empty()
        with pending_exchange_slot.container():
            live_response_placeholder, _unused_wrapper = _render_pending_exchange(
                conversation_mode=True,
                user_prompt=user_prompt,
            )
    else:
        latest_exchange_slot.empty()
        with latest_exchange_slot.container():
            live_response_placeholder, _unused_wrapper = _render_pending_exchange(
                conversation_mode=False,
                user_prompt=user_prompt,
            )
    if stream_output:
        live_response_placeholder.caption("No tokens received yet.")
    else:
        live_response_placeholder.caption("Streaming disabled. The full response will appear when generation completes.")

    def emit_progress(stage: str, progress_value: float, message: str) -> None:
        LOGGER.info("ui-progress [%s] %s", stage, message)
        run_status.write(message)
        run_progress.progress(min(max(progress_value, 0.0), 1.0), text=message)

    def emit_partial_text(text: str) -> None:
        preview = text.strip() or "Receiving generated tokens..."
        live_response_placeholder.markdown(f"**Live response**\n\n{preview}")

    started_at = perf_counter()

    try:
        emit_progress("start", 0.02, "Preparing request...")
        if uploaded_file is not None:
            uploaded_path = persist_upload(uploaded_file)
        if ability is Ability.VIDEO_TO_TEXT:
            if uploaded_path is None:
                raise ValueError("Upload a video before running video-to-text mode.")
            emit_progress("video", 0.10, "Extracting representative video frames...")
            frame_paths = extract_video_frames(uploaded_path)

        result = sandbox.run(
            ability=ability,
            user_prompt=user_prompt,
            uploaded_path=uploaded_path,
            frame_paths=frame_paths,
            prior_messages=model_history if conversation_mode else None,
            progress_callback=emit_progress,
            token_callback=emit_partial_text if stream_output else None,
        )
        elapsed_seconds = perf_counter() - started_at
        run_status.update(label="Sandbox run complete.", state="complete", expanded=False)
        live_response_placeholder.empty()

        if conversation_mode:
            model_history.append({"role": "user", "content": [{"type": "text", "text": result.prompt_used}]})
            model_history.append({"role": "assistant", "content": [{"type": "text", "text": result.response_text}]})
            ui_history.append({"role": "user", "content": user_prompt})
            ui_history.append({"role": "assistant", "content": result.response_text})
            pending_exchange_slot.empty()
            conversation_history_slot.empty()
            with conversation_history_slot.container():
                _render_conversation_history(ui_history)
        else:
            latest_exchange_slot.empty()
            with latest_exchange_slot.container():
                _render_latest_exchange(user_prompt, result.response_text)

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
                    "ability": ability.value,
                    "persona": preset_name,
                    "system_prompt": system_prompt,
                    "conversation_mode": conversation_mode,
                    "conversation_turn_count": len(ui_history) // 2 if conversation_mode else 0,
                    "enable_thinking": enable_thinking,
                    "stream_output": stream_output,
                    "max_new_tokens": max_new_tokens,
                    "temperature": config.generation.temperature,
                    "top_p": config.generation.top_p,
                    "top_k": config.generation.top_k,
                    "uploaded_file": uploaded_file.name if uploaded_file is not None else None,
                    "sampled_frame_count": len(frame_paths) if frame_paths else 0,
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


if __name__ == "__main__":
    main()