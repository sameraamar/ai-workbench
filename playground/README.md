# Playground

Standalone experiments and tools that aren't part of either deployable project.

- **gemma4_text_demo.py** — Direct Hugging Face Gemma 4 text inference demo. No project dependencies. Run with any venv that has `torch` and `transformers`.
- **benchmark_runner.py** — CLI harness for running benchmark scenarios against the model-serving API.
- **concurrency_simulation.py** — E2B vs E4B serving capacity simulation.

These scripts use whichever Python environment is active. They depend on `torch`, `transformers`, and optionally the `gemma_serving` package from `model-serving/`.
