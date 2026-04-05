# Running the Model Serving API

Four supported execution environments, ordered from simplest to most capable.

---

## Comparison

| | Windows (native) | WSL / Ubuntu | Linux (native) | vLLM |
|---|---|---|---|---|
| Setup effort | Lowest | Low | Low | High |
| `flash-attn` | ❌ No wheels | ✅ | ✅ | ✅ built-in |
| Long-sequence speed | Baseline (SDPA) | +10–30% | +10–30% | +50–100% |
| Mistral support | ⚠️ Transformers only | ⚠️ Transformers only | ⚠️ Transformers only | ✅ Official |
| Hot-reload on save | ⚠️ Polling | ✅ inotify | ✅ inotify | N/A |
| Recommended for | Development | Development + inference | CI / headless server | Mistral / production |

---

## Option 1 — Windows (PowerShell)

### Prerequisites

- Python 3.11 installed, `venv\` created at repo root
- CUDA drivers installed (RTX 3090)

### One-time setup

```powershell
# From repo root
cd model-serving
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 `
    --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
Copy-Item .env.example .env   # then edit .env for your setup
```

### Run

```powershell
cd model-serving
.\start_server.ps1                  # 127.0.0.1:8000, hot-reload on
.\start_server.ps1 --Port 8001      # custom port
.\start_server.ps1 -NoReload        # disable hot-reload
```

### Notes

- `flash-attn` has no prebuilt Windows wheels. `MODEL_FLASH_ATTENTION=1` silently falls
  back to PyTorch SDPA — functional but ~10–30% slower on long sequences.
- The `venv\` folder lives under the repo root on NTFS. Do NOT create a second venv
  under `/mnt/c/` from WSL — pip's `.pyc` write fails on the 9P NTFS driver.

---

## Option 2 — WSL (Ubuntu on Windows)

Run the model on your Windows GPU but from a Linux shell. Gains `flash-attn` and
inotify-based hot-reload without leaving your Windows machine.

### One-time setup

```bash
# 1. Create venv on the Linux filesystem (NOT under /mnt/c/)
python3 -m venv ~/venvs/ai-workbench
source ~/venvs/ai-workbench/bin/activate

# 2. PyTorch with CUDA (Linux wheels, same GPU via WSL2 passthrough)
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# 3. Project dependencies (requirements.txt stays on /mnt/c/)
pip install -r /mnt/c/Users/saaamar/source/repos/ai-workbench/model-serving/requirements.txt

# 4. flash-attn — Linux only, optional but recommended
pip install flash-attn --no-build-isolation

# 5. Reuse the Windows HuggingFace cache — avoids re-downloading large models
echo 'export HF_HOME=/mnt/c/Users/saaamar/.cache/huggingface' >> ~/.bashrc
source ~/.bashrc
```

### Run (every session)

```bash
source ~/venvs/ai-workbench/bin/activate
cd /mnt/c/Users/saaamar/source/repos/ai-workbench/model-serving
./start_server.sh                   # 127.0.0.1:8000, hot-reload on
./start_server.sh --port 8001       # custom port
./start_server.sh --no-reload       # disable hot-reload
```

### Notes

- **CRLF bug**: `.env` has Windows CRLF line endings. `start_server.sh` strips trailing
  `\r` automatically. If you load `.env` yourself (e.g. `source .env`) you will get
  `device_map=auto\r` → `ValueError`. Use the script instead.
- The venv **must** live under the Linux filesystem (`~/...`), not `/mnt/c/`. Pip
  writing `.pyc` files to NTFS via the 9P driver raises `AssertionError`.

---

## Option 3 — Linux (native)

Same as WSL but without the `/mnt/c/` path indirection.

### One-time setup

```bash
# Clone the repo or pull latest
cd ~/repos/ai-workbench/model-serving

python3 -m venv .venv
source .venv/bin/activate

pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install flash-attn --no-build-isolation   # recommended

cp .env.example .env   # edit for your setup
```

### Run

```bash
source .venv/bin/activate
./start_server.sh
```

---

## Option 4 — vLLM (recommended for Mistral)

Mistral AI officially states that the `transformers` integration is "vibe-checked only".
For reliable Mistral inference (correct tokenization, vision inputs, generation numerics)
use vLLM. Gemma 4 also runs faster under vLLM.

### Prerequisites

- Linux or WSL (vLLM has no Windows support)
- `pip install vllm` (requires separate setup; large install)

### Run

```bash
vllm serve mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
    --tokenizer_mode mistral \
    --config_format mistral \
    --load_format mistral \
    --limit_mm_per_prompt "image=10" \
    --host 127.0.0.1 \
    --port 8000
```

For Gemma 4:
```bash
vllm serve google/gemma-4-E2B-it \
    --host 127.0.0.1 \
    --port 8000
```

The vLLM server exposes the same OpenAI-compatible `/v1/chat/completions` and
`/v1/models` endpoints — the UI and `serving_client.py` work against it unchanged.

> **Status**: vLLM integration is tracked in `docs/tasks.md`. The current FastAPI
> server (`model-serving/`) uses `transformers` and works for Gemma. Mistral is
> in the UI's `DISABLED_LABELS` until the vLLM path is wired up.

---

## Verify the server is running

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
```

---

## Key `.env` variables

| Variable | Default | Effect |
|---|---|---|
| `MODEL_ID` | `google/gemma-4-E2B-it` | HuggingFace model to load |
| `MODEL_QUANTIZE_4BIT` | `0` | `1` = 4-bit NF4 (required for 24B+ on 24 GB VRAM) |
| `MODEL_TORCH_COMPILE` | `1` | `0` = disable (recommended; overhead on short sequences) |
| `MODEL_FLASH_ATTENTION` | `1` | Use flash-attn if installed, else SDPA fallback |
| `MODEL_DEVICE_MAP` | `auto` | GPU placement strategy |
| `MODEL_MAX_INPUT_TOKENS` | `8192` | Truncation cap before generation to prevent OOM |
| `MODEL_GATEWAY` | `model` | `model` = real inference; `stub` = empty responses for API tests |
