# Purpose

This architecture decision record captures the initial structural choices for the project.
It exists so later contributors can understand why the starter stack and capability boundaries were chosen.
AI agents should read this before changing the top-level architecture.

# Maintenance Instructions

Update this file only when the related decision changes or is superseded.
Humans or AI may append a new ADR for major architectural changes rather than rewriting history.
This file complements [docs/design/design.md](../design/design.md).

## Decision

Use a documentation-first Python project with a Streamlit UI, a reusable Gemma service module, and explicit separation between native Gemma workflows and simulated media-planning workflows.

## Status

Accepted

## Context

The repository starts empty and needs both persistent documentation and a runnable prototype.
The task requests a UI with multiple ability choices, including modes that Gemma 4 does not natively render as media outputs.

## Consequences

### Positive

- Fastest path to a usable local prototype
- Clear separation between model access and UI logic
- Honest handling of unsupported generation modes
- Easy future replacement of simulated modes with external backends

### Negative

- Streamlit is suitable for prototyping but not necessarily the final product surface
- Real multimodal execution quality depends on local hardware and model availability
- Video-to-text is an approximation through frame sampling in the starter version

## Alternatives Considered

- Building a custom frontend plus API first
  - Rejected because it adds more scaffolding than needed for the first milestone

- Pretending Gemma natively supports text-to-image, text-to-video, and text-to-audio
  - Rejected because it would be technically misleading