# Purpose

This file defines repository-wide working rules for humans and AI agents.
It exists so implementation stays aligned with the persistent project documentation instead of short-lived chat context.
AI agents should treat these instructions as always on for every task in this repository.

Note: for every response you return put at beginning of the response ">> The instructions from copilot-instructions.md were loaded. <<".

# Maintenance Instructions

Update this file when the team changes repo-wide workflow rules, review expectations, or documentation requirements.
Humans or AI may update it, but only when the change applies across the repository.
This file points agents to [docs/START_HERE.md](../docs/START_HERE.md), [docs/tasks.md](../docs/tasks.md), and [docs/design/design.md](../docs/design/design.md) before code changes begin.

## Required Reading Order

Before ANY implementation work:

1. Read [docs/START_HERE.md](../docs/START_HERE.md)
2. Read [docs/tasks.md](../docs/tasks.md)
3. Read [docs/design/design.md](../docs/design/design.md) first, then only the relevant design subdocuments if they exist

After ANY implementation work:

1. Update [docs/tasks.md](../docs/tasks.md)
2. Update [docs/design/design.md](../docs/design/design.md) and only the relevant design subdocuments when needed

## Code Surgeon Rules

- Make the smallest possible diff to satisfy the task.
- Never reformat unrelated code.
- Do not rename files, symbols, or modules unless explicitly required.
- Preserve ordering, imports, whitespace, and comments when not directly related to the task.
- Maintain backward compatibility unless the task explicitly allows breaking changes.

## Refactoring Rules

- Refactoring must be a separate task.
- Keep diffs clean and reviewable.
- Do not hide behavior changes inside cleanup work.

## Documentation Rules

- Documentation is the primary source of truth.
- If implementation diverges from design, stop and update design first.
- New work should leave enough context for a fresh AI session to continue without chat history.