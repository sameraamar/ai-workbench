# Purpose

This document stores research findings, external facts, and early rationale that inform the project.
It exists so future AI sessions can reuse validated context without re-reading the same source material from scratch.
AI agents should update this when they confirm facts that affect implementation choices.

# Maintenance Instructions

Update this file when new external research changes model assumptions, capability boundaries, or best practices.
Humans or AI may update it.
Keep entries concise and focused on facts that materially affect design or implementation.

## Current Research Notes

- Gemma 4 E2B and E4B support text, image, and audio input with text output.
- Gemma 4 can be used through Hugging Face Transformers with a chat-style message format.
- Standardized sampling guidance from the provided model card and article:
  - temperature = 1.0
  - top_p = 0.95
  - top_k = 64
- Gemma 4 is strong at reasoning, coding, multimodal understanding, and planning.
- Gemma 4 does not natively generate images, video, or audio assets. It generates text responses.
- Video understanding can be approached by sampling frames and asking for scene analysis.
- Best modality ordering for multimodal prompts is media first, then text.
- Raw source material copied from the Gemma 4 model card is archived in [docs/research/sources/gemma4-model-card.md](./sources/gemma4-model-card.md).
- Production-serving research for model choice, concurrency, latency, and cost planning is tracked in [docs/research/gemma4-serving-evaluation.md](./gemma4-serving-evaluation.md).

## Chosen Sandbox Rationale

The selected sandbox usage is the Multimodal Situation Room.

Why this is a good fit:

- It centers analysis and planning, which are Gemma 4 strengths.
- It supports a game-like UI without lying about unsupported media synthesis.
- It lets the app expose all requested mode choices while clearly labeling which ones are native versus simulated.
- It creates a future integration seam for real text-to-image, text-to-video, and text-to-audio backends later.