from __future__ import annotations

import ctypes
import logging
import os
import sys
from threading import Lock, Thread
from time import perf_counter
from typing import Any, Callable

import torch

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from transformers import AutoModelForCausalLM, AutoModelForMultimodalLM, AutoProcessor, Gemma4ForConditionalGeneration, TextIteratorStreamer

from gemma_serving.config import ServingConfig, GenerationSettings

LOGGER = logging.getLogger(__name__)
ProgressCallback = Callable[[str, float, str], None]
TokenCallback = Callable[[str], None]


class GenerationResult(dict):
    text: str
    input_token_count: int | None
    output_token_count: int | None
    total_token_count: int | None


class GemmaService:
    def __init__(self, config: ServingConfig) -> None:
        self._config = config
        self._processor = None
        self._text_model = None
        self._multimodal_model = None
        self._lock = Lock()

    def is_model_loaded(self) -> bool:
        return self._processor is not None and (self._text_model is not None or self._multimodal_model is not None)

    def ensure_loaded(self) -> None:
        """Pre-load the text runtime so the first generate call is fast."""
        self._get_text_runtime()

    def generate(
        self,
        messages: list[dict[str, Any]],
        settings: GenerationSettings | None = None,
        progress_callback: ProgressCallback | None = None,
        token_callback: TokenCallback | None = None,
    ) -> dict[str, Any]:
        active_settings = settings or self._config.generation
        if _is_text_only(messages):
            return self._generate_text(messages, active_settings, progress_callback, token_callback)
        return self._generate_multimodal(messages, active_settings, progress_callback)

    def _generate_text(
        self,
        messages: list[dict[str, Any]],
        settings: GenerationSettings,
        progress_callback: ProgressCallback | None = None,
        token_callback: TokenCallback | None = None,
    ) -> dict[str, Any]:
        service_started_at = perf_counter()
        memory_before = _capture_memory_snapshot()
        runtime_started_at = perf_counter()
        processor, model = self._get_text_runtime(progress_callback)
        runtime_load_seconds = perf_counter() - runtime_started_at

        prepare_started_at = perf_counter()
        self._emit(progress_callback, "prepare", 0.70, "Preparing Gemma text prompt...")
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=settings.enable_thinking,
        )
        model_device = _resolve_model_device(model)
        inputs = processor(text=text, return_tensors="pt").to(model_device)
        input_len = inputs["input_ids"].shape[-1]
        prompt_char_count = len(text)
        memory_after_runtime = _capture_memory_snapshot(model_device)
        _reset_peak_memory_stats(model_device)
        prepare_seconds = perf_counter() - prepare_started_at
        generation_started_at = perf_counter()
        if settings.stream_output:
            response = self._generate_text_streaming(
                processor=processor,
                model=model,
                inputs=inputs,
                settings=settings,
                progress_callback=progress_callback,
                token_callback=token_callback,
            )
        else:
            response = self._generate_text_one_shot(
                processor=processor,
                model=model,
                inputs=inputs,
                input_len=input_len,
                settings=settings,
                progress_callback=progress_callback,
            )
        generation_seconds = perf_counter() - generation_started_at

        decode_started_at = perf_counter()
        self._emit(progress_callback, "decode", 0.96, "Finalizing response...")
        parsed = _parse_response(processor, response)
        input_token_count = input_len
        output_token_count = _count_text_tokens(processor, parsed)
        response_char_count = len(parsed)
        decode_seconds = perf_counter() - decode_started_at
        memory_after_run = _capture_memory_snapshot(model_device)
        metadata = _build_generation_metadata(
            prompt_char_count=prompt_char_count,
            response_char_count=response_char_count,
            runtime_load_seconds=runtime_load_seconds,
            prepare_seconds=prepare_seconds,
            generation_seconds=generation_seconds,
            decode_seconds=decode_seconds,
            total_service_seconds=perf_counter() - service_started_at,
            output_token_count=output_token_count,
            memory_before=memory_before,
            memory_after_runtime=memory_after_runtime,
            memory_after_run=memory_after_run,
            model_device=model_device,
        )
        self._emit(progress_callback, "complete", 1.0, "Gemma response ready.")
        return {
            "text": parsed,
            "input_token_count": input_token_count,
            "output_token_count": output_token_count,
            "total_token_count": _sum_token_counts(input_token_count, output_token_count),
            "metadata": metadata,
        }

    def _generate_text_streaming(
        self,
        *,
        processor: Any,
        model: Any,
        inputs: Any,
        settings: GenerationSettings,
        progress_callback: ProgressCallback | None,
        token_callback: TokenCallback | None,
    ) -> str:
        streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=False)
        generation_error: list[Exception] = []

        def run_generation() -> None:
            try:
                with torch.inference_mode():
                    model.generate(
                        **inputs,
                        do_sample=True,
                        temperature=settings.temperature,
                        top_p=settings.top_p,
                        top_k=settings.top_k,
                        max_new_tokens=settings.max_new_tokens,
                        streamer=streamer,
                    )
            except Exception as error:  # pragma: no cover - runtime path
                generation_error.append(error)

        self._emit(progress_callback, "generate", 0.84, "Running text generation in streaming mode...")
        worker = Thread(target=run_generation, daemon=True)
        worker.start()

        streamed_response = ""
        first_token_seen = False
        for chunk in streamer:
            if not first_token_seen:
                self._emit(progress_callback, "stream", 0.90, "Receiving generated tokens...")
                first_token_seen = True
            streamed_response += chunk
            if token_callback is not None:
                token_callback(streamed_response)

        worker.join()
        if generation_error:
            raise generation_error[0]
        return streamed_response

    def _generate_text_one_shot(
        self,
        *,
        processor: Any,
        model: Any,
        inputs: Any,
        input_len: int,
        settings: GenerationSettings,
        progress_callback: ProgressCallback | None,
    ) -> str:
        self._emit(progress_callback, "generate", 0.84, "Running text generation in one-shot mode...")
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                do_sample=True,
                temperature=settings.temperature,
                top_p=settings.top_p,
                top_k=settings.top_k,
                max_new_tokens=settings.max_new_tokens,
            )
        return processor.decode(outputs[0][input_len:], skip_special_tokens=False)

    def _generate_multimodal(
        self,
        messages: list[dict[str, Any]],
        settings: GenerationSettings,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        service_started_at = perf_counter()
        memory_before = _capture_memory_snapshot()
        runtime_started_at = perf_counter()
        processor, model = self._get_multimodal_runtime(progress_callback)
        runtime_load_seconds = perf_counter() - runtime_started_at

        prepare_started_at = perf_counter()
        self._emit(progress_callback, "prepare", 0.70, "Preparing Gemma inputs...")
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        model_device = _resolve_model_device(model)
        inputs = inputs.to(model_device)
        input_len = inputs["input_ids"].shape[-1]
        prompt_char_count = _extract_message_character_count(messages)
        memory_after_runtime = _capture_memory_snapshot(model_device)
        _reset_peak_memory_stats(model_device)
        prepare_seconds = perf_counter() - prepare_started_at

        generation_started_at = perf_counter()
        self._emit(progress_callback, "generate", 0.84, "Running generation...")
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=settings.temperature,
            top_p=settings.top_p,
            top_k=settings.top_k,
            max_new_tokens=settings.max_new_tokens,
        )
        generation_seconds = perf_counter() - generation_started_at

        decode_started_at = perf_counter()
        self._emit(progress_callback, "decode", 0.96, "Decoding response...")
        response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
        parsed = _parse_response(processor, response)
        input_token_count = input_len
        output_token_count = int(outputs[0].shape[-1] - input_len)
        response_char_count = len(parsed)
        decode_seconds = perf_counter() - decode_started_at
        memory_after_run = _capture_memory_snapshot(model_device)
        metadata = _build_generation_metadata(
            prompt_char_count=prompt_char_count,
            response_char_count=response_char_count,
            runtime_load_seconds=runtime_load_seconds,
            prepare_seconds=prepare_seconds,
            generation_seconds=generation_seconds,
            decode_seconds=decode_seconds,
            total_service_seconds=perf_counter() - service_started_at,
            output_token_count=output_token_count,
            memory_before=memory_before,
            memory_after_runtime=memory_after_runtime,
            memory_after_run=memory_after_run,
            model_device=model_device,
        )
        self._emit(progress_callback, "complete", 1.0, "Gemma response ready.")
        return {
            "text": parsed,
            "input_token_count": input_token_count,
            "output_token_count": output_token_count,
            "total_token_count": _sum_token_counts(input_token_count, output_token_count),
            "metadata": metadata,
        }

    def _get_text_runtime(self, progress_callback: ProgressCallback | None = None):
        if self._processor is not None and self._text_model is not None:
            self._emit(progress_callback, "runtime", 0.62, "Text model already loaded in memory.")
            return self._processor, self._text_model

        with self._lock:
            if self._processor is None or self._text_model is None:
                dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
                self._emit(progress_callback, "runtime", 0.06, "Checking Gemma runtime state...")
                self._ensure_processor(progress_callback)
                quant_label = " (4-bit quantized)" if self._config.quantize_4bit else ""
                self._emit(progress_callback, "model", 0.34, f"Loading Gemma text weights{quant_label}. The first run may download several GB and take a few minutes...")
                self._text_model = AutoModelForCausalLM.from_pretrained(
                    self._config.model_id,
                    **_build_model_load_kwargs(dtype, quantize_4bit=self._config.quantize_4bit, force_download=self._config.force_download),
                )
                self._text_model.eval()
                self._emit(progress_callback, "model", 0.62, "Text runtime loaded.")
        return self._processor, self._text_model

    def _get_multimodal_runtime(self, progress_callback: ProgressCallback | None = None):
        if self._processor is not None and self._multimodal_model is not None:
            self._emit(progress_callback, "runtime", 0.62, "Multimodal model already loaded in memory.")
            return self._processor, self._multimodal_model

        with self._lock:
            if self._processor is None or self._multimodal_model is None:
                dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
                self._emit(progress_callback, "runtime", 0.06, "Checking Gemma runtime state...")
                self._ensure_processor(progress_callback)
                quant_label = " (4-bit quantized)" if self._config.quantize_4bit else ""
                self._emit(progress_callback, "model", 0.34, f"Loading Gemma multimodal weights{quant_label}. The first run may download several GB and take a few minutes...")
                self._multimodal_model = _load_multimodal_model(
                    self._config.model_id, dtype,
                    quantize_4bit=self._config.quantize_4bit,
                    force_download=self._config.force_download,
                )
                self._multimodal_model.eval()
                self._emit(progress_callback, "model", 0.62, "Multimodal runtime loaded.")
        return self._processor, self._multimodal_model

    def _ensure_processor(self, progress_callback: ProgressCallback | None = None) -> None:
        if self._processor is None:
            self._emit(
                progress_callback,
                "processor",
                0.16,
                f"Loading processor for {self._config.model_id}...",
            )
            self._processor = AutoProcessor.from_pretrained(self._config.model_id)

    def _emit(
        self,
        progress_callback: ProgressCallback | None,
        stage: str,
        progress_value: float,
        message: str,
    ) -> None:
        LOGGER.info("[%s] %s", stage, message)
        if progress_callback is not None:
            progress_callback(stage, progress_value, message)


def _resolve_model_device(model: Any) -> torch.device:
    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        for candidate in hf_device_map.values():
            if candidate in {"disk", "meta", None}:
                continue
            return torch.device(candidate)
    try:
        for parameter in model.parameters():
            if parameter.device.type != "meta":
                return parameter.device
    except StopIteration:
        pass
    return torch.device("cpu")


def _load_multimodal_model(model_id: str, dtype: torch.dtype, *, quantize_4bit: bool = False, force_download: bool = False):
    model_loaders = (
        Gemma4ForConditionalGeneration,
        AutoModelForMultimodalLM,
    )

    last_error: Exception | None = None
    for model_loader in model_loaders:
        try:
            LOGGER.info("Trying model loader: %s", model_loader.__name__)
            return model_loader.from_pretrained(
                model_id,
                **_build_model_load_kwargs(dtype, quantize_4bit=quantize_4bit, force_download=force_download),
            )
        except Exception as error:  # pragma: no cover - fallback path
            last_error = error
            LOGGER.warning("Model loader %s failed: %s", model_loader.__name__, error)

    if last_error is not None:
        raise last_error
    raise RuntimeError("No compatible Gemma model loader was available.")


def _is_text_only(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") != "text":
                    return False
    return True


def _parse_response(processor: Any, response: str) -> str:
    if hasattr(processor, "parse_response"):
        try:
            parsed = processor.parse_response(response)
            extracted = _extract_text_from_parsed(parsed)
            if extracted:
                return extracted
        except Exception as error:  # pragma: no cover - parser fallback
            LOGGER.warning("Gemma response parsing failed: %s", error)
    return response.strip()


def _extract_text_from_parsed(parsed: Any) -> str:
    if isinstance(parsed, str):
        return parsed.strip()
    if isinstance(parsed, dict):
        for key in ("response", "text", "content", "final_response", "answer"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        final = parsed.get("final")
        if isinstance(final, str) and final.strip():
            return final.strip()
    return ""


def _count_text_tokens(processor: Any, text: str) -> int | None:
    stripped = text.strip()
    if not stripped:
        return 0

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        try:
            encoded = tokenizer(stripped, add_special_tokens=False, return_tensors="pt")
            return int(encoded["input_ids"].shape[-1])
        except Exception as error:  # pragma: no cover - tokenizer fallback
            LOGGER.warning("Tokenizer-based token counting failed: %s", error)

    try:
        encoded = processor(text=stripped, return_tensors="pt")
        return int(encoded["input_ids"].shape[-1])
    except Exception as error:  # pragma: no cover - processor fallback
        LOGGER.warning("Processor-based token counting failed: %s", error)
        return None


def _sum_token_counts(input_token_count: int | None, output_token_count: int | None) -> int | None:
    if input_token_count is None or output_token_count is None:
        return None
    return input_token_count + output_token_count


def _build_generation_metadata(
    *,
    prompt_char_count: int,
    response_char_count: int,
    runtime_load_seconds: float,
    prepare_seconds: float,
    generation_seconds: float,
    decode_seconds: float,
    total_service_seconds: float,
    output_token_count: int | None,
    memory_before: dict[str, Any],
    memory_after_runtime: dict[str, Any],
    memory_after_run: dict[str, Any],
    model_device: torch.device,
) -> dict[str, Any]:
    return {
        "prompt_char_count": prompt_char_count,
        "response_char_count": response_char_count,
        "timings": {
            "runtime_load_seconds": round(runtime_load_seconds, 6),
            "prepare_seconds": round(prepare_seconds, 6),
            "generation_seconds": round(generation_seconds, 6),
            "decode_seconds": round(decode_seconds, 6),
            "service_seconds": round(total_service_seconds, 6),
        },
        "output_tokens_per_second": _calculate_tokens_per_second(output_token_count, generation_seconds),
        "memory": _summarize_memory_usage(memory_before, memory_after_runtime, memory_after_run, model_device),
    }


def _extract_message_character_count(messages: list[dict[str, Any]]) -> int:
    total = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            total += len(content)
            continue
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                for key in ("text", "url", "audio"):
                    value = block.get(key)
                    if isinstance(value, str):
                        total += len(value)
    return total


def _calculate_tokens_per_second(output_token_count: int | None, generation_seconds: float) -> float | None:
    if output_token_count is None or generation_seconds <= 0:
        return None
    return round(output_token_count / generation_seconds, 6)


def _capture_memory_snapshot(model_device: torch.device | None = None) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "process_rss_bytes": _get_process_rss_bytes(),
    }
    device = model_device
    if device is None and torch.cuda.is_available():
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
        snapshot.update(
            {
                "device": str(device),
                "cuda_allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "cuda_reserved_bytes": int(torch.cuda.memory_reserved(device)),
                "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }
        )
    return snapshot


def _reset_peak_memory_stats(model_device: torch.device) -> None:
    if model_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(model_device)


def _summarize_memory_usage(
    memory_before: dict[str, Any],
    memory_after_runtime: dict[str, Any],
    memory_after_run: dict[str, Any],
    model_device: torch.device,
) -> dict[str, Any]:
    memory: dict[str, Any] = {
        "device": str(model_device),
        "process_rss_mb_before": _bytes_to_mb(memory_before.get("process_rss_bytes")),
        "process_rss_mb_after_runtime": _bytes_to_mb(memory_after_runtime.get("process_rss_bytes")),
        "process_rss_mb_after_run": _bytes_to_mb(memory_after_run.get("process_rss_bytes")),
        "process_rss_delta_mb": _delta_mb(
            memory_before.get("process_rss_bytes"),
            memory_after_run.get("process_rss_bytes"),
        ),
    }
    if model_device.type == "cuda":
        memory.update(
            {
                "cuda_allocated_mb_after_runtime": _bytes_to_mb(memory_after_runtime.get("cuda_allocated_bytes")),
                "cuda_allocated_mb_after_run": _bytes_to_mb(memory_after_run.get("cuda_allocated_bytes")),
                "cuda_reserved_mb_after_runtime": _bytes_to_mb(memory_after_runtime.get("cuda_reserved_bytes")),
                "cuda_reserved_mb_after_run": _bytes_to_mb(memory_after_run.get("cuda_reserved_bytes")),
                "cuda_peak_allocated_mb": _bytes_to_mb(memory_after_run.get("cuda_peak_allocated_bytes")),
                "cuda_peak_reserved_mb": _bytes_to_mb(memory_after_run.get("cuda_peak_reserved_bytes")),
            }
        )
    return memory


def _bytes_to_mb(value: Any) -> float | None:
    if value is None:
        return None
    return round(float(value) / (1024 * 1024), 3)


def _delta_mb(before: Any, after: Any) -> float | None:
    if before is None or after is None:
        return None
    return round((float(after) - float(before)) / (1024 * 1024), 3)


def _get_process_rss_bytes() -> int | None:
    if sys.platform == "win32":
        return _get_process_rss_windows()
    return _get_process_rss_posix()


def _get_process_rss_windows() -> int | None:
    class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    counters = PROCESS_MEMORY_COUNTERS_EX()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
    process = ctypes.windll.kernel32.GetCurrentProcess()
    success = ctypes.windll.psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb)
    if not success:
        return None
    return int(counters.WorkingSetSize)


def _get_process_rss_posix() -> int | None:
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return int(rss)
        return int(rss * 1024)
    except Exception:
        return None


def _build_model_load_kwargs(dtype: torch.dtype, *, quantize_4bit: bool = False, force_download: bool = False) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"low_cpu_mem_usage": True}
    if force_download:
        kwargs["force_download"] = True
    if quantize_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["dtype"] = dtype
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    return kwargs