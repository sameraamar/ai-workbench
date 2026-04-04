from __future__ import annotations

import ctypes
import os
import sys
from time import perf_counter

import torch
from transformers import AutoModelForCausalLM, AutoProcessor


MODEL_ID = os.getenv("GEMMA_MODEL_ID", "google/gemma-4-E2B-it")
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = 64
MAX_NEW_TOKENS = 256
SYSTEM_PROMPT = "You are a helpful assistant."


def main() -> None:
    print("=" * 60)
    print("🚀 GEMMA 4 STANDALONE DEMO - CUDA TEST")
    print("=" * 60)
    
    # CUDA Detection and Logging
    print("\n🔍 CHECKING GPU/CUDA AVAILABILITY:")
    print(f"   PyTorch Version: {torch.__version__}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   ✅ GPU Count: {torch.cuda.device_count()}")
        print(f"   ✅ GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"   ✅ CUDA Version: {torch.version.cuda}")
        print(f"   ✅ Model will use: GPU acceleration")
        device_info = "🚀 CUDA GPU"
    else:
        print("   ❌ GPU Count: 0")
        print("   ❌ GPU Name: N/A") 
        print("   ❌ CUDA Version: Not Available")
        print("   ⚠️  Model will use: CPU (SLOW!)")
        device_info = "🐌 CPU ONLY"
    
    print(f"\n🔧 DEVICE MODE: {device_info}")
    print("=" * 60)
    
    load_started_at = perf_counter()
    memory_before_load = capture_memory_snapshot()
    print(f"\n📦 Loading processor for {MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    print(f"🧠 Loading model for {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    
    # Post-load device confirmation
    actual_device = next(model.parameters()).device
    print(f"✅ Model loaded on device: {actual_device}")
    
    load_elapsed_seconds = perf_counter() - load_started_at
    memory_after_load = capture_memory_snapshot(actual_device)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("Gemma 4 multi-turn demo is ready.")
    print('Type your message and press Enter. Type "END" to finish the conversation.')
    print(f"Startup load time: {load_elapsed_seconds:.2f} seconds")
    print(
        "Startup memory: "
        f"RSS delta {format_optional_count(delta_mb(memory_before_load.get('process_rss_bytes'), memory_after_load.get('process_rss_bytes')))} MB"
    )

    while True:
        prompt = input('\nYour message (type "END" to quit): ').strip()
        if not prompt:
            print("Please enter a message or type END to stop.")
            continue
        if prompt.upper() == "END":
            print("Ending conversation.")
            break

        messages.append({"role": "user", "content": prompt})

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = processor(text=text, return_tensors="pt").to(actual_device)
        input_len = inputs["input_ids"].shape[-1]
        prompt_char_count = len(text)
        memory_before_generation = capture_memory_snapshot(actual_device)
        reset_peak_memory_stats(actual_device)

        print("Generating response...")
        started_at = perf_counter()
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        elapsed_seconds = perf_counter() - started_at
        memory_after_generation = capture_memory_snapshot(actual_device)

        response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
        parsed_response = None

        if hasattr(processor, "parse_response"):
            try:
                parsed_response = processor.parse_response(response)
                if isinstance(parsed_response, dict):
                    response = (
                        parsed_response.get("response")
                        or parsed_response.get("text")
                        or parsed_response.get("content")
                        or parsed_response.get("final_response")
                        or response
                    )
                elif isinstance(parsed_response, str):
                    response = parsed_response
            except Exception:
                pass

        final_response = str(response).strip()
        output_token_count = count_text_tokens(processor, final_response)
        total_token_count = sum_token_counts(input_len, output_token_count)
        response_char_count = len(final_response)
        generation_tokens_per_second = calculate_tokens_per_second(output_token_count, elapsed_seconds)
        messages.append({"role": "assistant", "content": final_response})

        print("\nGemma 4 response:\n")
        print(final_response)
        print(f"\nResponse time: {elapsed_seconds:.2f} seconds")
        print(f"Input tokens: {input_len}")
        print(f"Output tokens: {format_optional_count(output_token_count)}")
        print(f"Total tokens: {format_optional_count(total_token_count)}")
        print(f"Prompt chars: {prompt_char_count}")
        print(f"Response chars: {response_char_count}")
        print(f"Generation tok/s: {format_optional_count(generation_tokens_per_second)}")
        print(
            "Memory: "
            f"RSS delta {format_optional_count(delta_mb(memory_before_generation.get('process_rss_bytes'), memory_after_generation.get('process_rss_bytes')))} MB"
        )
        if actual_device.type == "cuda":
            print(
                "VRAM: "
                f"allocated {format_optional_count(bytes_to_mb(memory_after_generation.get('cuda_allocated_bytes')))} MB | "
                f"peak {format_optional_count(bytes_to_mb(memory_after_generation.get('cuda_peak_allocated_bytes')))} MB"
            )


def count_text_tokens(processor: AutoProcessor, text: str) -> int | None:
    stripped = text.strip()
    if not stripped:
        return 0

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        try:
            encoded = tokenizer(stripped, add_special_tokens=False, return_tensors="pt")
            return int(encoded["input_ids"].shape[-1])
        except Exception:
            pass

    try:
        encoded = processor(text=stripped, return_tensors="pt")
        return int(encoded["input_ids"].shape[-1])
    except Exception:
        return None


def sum_token_counts(input_token_count: int | None, output_token_count: int | None) -> int | None:
    if input_token_count is None or output_token_count is None:
        return None
    return input_token_count + output_token_count


def calculate_tokens_per_second(output_token_count: int | None, elapsed_seconds: float) -> float | None:
    if output_token_count is None or elapsed_seconds <= 0:
        return None
    return round(output_token_count / elapsed_seconds, 3)


def capture_memory_snapshot(model_device: torch.device | None = None) -> dict[str, int | str | None]:
    snapshot: dict[str, int | str | None] = {
        "process_rss_bytes": get_process_rss_bytes(),
    }
    if model_device is not None and model_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(model_device)
        snapshot.update(
            {
                "device": str(model_device),
                "cuda_allocated_bytes": int(torch.cuda.memory_allocated(model_device)),
                "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(model_device)),
            }
        )
    return snapshot


def reset_peak_memory_stats(model_device: torch.device) -> None:
    if model_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(model_device)


def bytes_to_mb(value: int | None) -> float | None:
    if value is None:
        return None
    return round(value / (1024 * 1024), 3)


def delta_mb(before: int | None, after: int | None) -> float | None:
    if before is None or after is None:
        return None
    return round((after - before) / (1024 * 1024), 3)


def get_process_rss_bytes() -> int | None:
    if sys.platform == "win32":
        return get_process_rss_windows()
    return get_process_rss_posix()


def get_process_rss_windows() -> int | None:
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


def get_process_rss_posix() -> int | None:
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return int(rss)
        return int(rss * 1024)
    except Exception:
        return None


def format_optional_count(value: int | float | None) -> str:
    return str(value) if value is not None else "unavailable"


if __name__ == "__main__":
    main()