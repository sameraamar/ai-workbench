# Purpose

This is the project entrypoint document.
It exists to give humans and AI agents a stable restart point for understanding the project, its current phase, and where authoritative decisions live.
AI agents should read this first at the start of every task.

# Maintenance Instructions

Update this file when the project phase changes, the source-of-truth files move, or the kickoff workflow needs to change.
Humans or AI may update it as part of normal project evolution.
This file summarizes and indexes the more detailed truth in [docs/tasks.md](./tasks.md) and [docs/design/design.md](./design/design.md).

## Project Snapshot

- Project name: To Be Discovered
- Working title: AI Workbench
- Project description: A local-first sandbox UI for exploring multimodal AI models through text, image, audio, and video understanding workflows, plus simulated media-generation planning workflows. The architecture is designed to support multiple models (Gemma 4, Llama, Phi, Mistral, etc.) without changing the UI or API contract.
- Target users / personas: To Be Discovered
- Primary business goal: To Be Discovered
- Technology preferences: Python, Streamlit, Hugging Face Transformers, PyTorch, FastAPI, httpx
- Constraints: Must encapsulate model access in a reusable module, must be honest about native model capabilities versus simulated workflows
- Expected deployment type: Prototype

## Repository Layout

The repo is split into three independent projects under a single git root:

| Folder | Purpose | Entry point | Tests |
|---|---|---|---|
| `model-serving/` | FastAPI model-serving backend (loads models, exposes `/generate` and job endpoints) | `uvicorn model_serving.app:app` | `model-serving/tests/` (24 tests) |
| `ui/` | Streamlit sandbox UI (calls model-serving over HTTP) | `streamlit run ui/app.py` | `ui/tests/` (4 tests) |
| `playground/` | Standalone demo and benchmark scripts | Individual `.py` files | — |

Each project has its own `requirements.txt`, `.env.example`, and `PYTHONPATH` root (`model-serving/src` or `ui/src`).

## Current Phase

Phase 2: Three-project architecture established. Model-serving and UI are decoupled.

## Source Of Truth

The current truth for project intent and implementation priority is:

1. [docs/tasks.md](./tasks.md)
2. [docs/design/design.md](./design/design.md)
3. [docs/benchmarks.md](./benchmarks.md) - Performance testing results and capacity planning guidance
4. Relevant design subdocuments if they are added later

## Restarting AI Sessions

Use this order every time a new AI session begins:

1. Read [docs/START_HERE.md](./START_HERE.md)
2. Read [docs/tasks.md](./tasks.md)
3. Read [docs/design/design.md](./design/design.md)
4. Read only the relevant design subdocuments if they exist
5. Summarize current intent, current phase, and the next minimal implementation step

## Starting A New Task

1. Confirm the task exists in [docs/tasks.md](./tasks.md), or add it under discovered work.
2. Read the design sections relevant to the task.
3. Propose a minimal, reviewable plan.
4. Implement the smallest viable diff.
5. Update tasks and design docs after the code change.

## Task Kickoff Script

Read docs/START_HERE.md, then docs/tasks.md,
then docs/design/design.md,
then the relevant design subdocuments if they exist.

Summarize the current project intent and phase.

Ask clarifying questions ONLY if required.

Then propose a minimal, reviewable implementation plan.