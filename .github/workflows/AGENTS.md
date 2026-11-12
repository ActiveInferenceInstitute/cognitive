---
title: CI workflow instructions
type: agents
status: stable
updated: 2026-11-03
---

# AGENTS.md — `.github/workflows/`

- Keep the pinned `pandoc` and `pandoc-crossref` versions a pair; a crossref
  binary loads only under the Pandoc it was built against.
- A skipped render or an unprovisioned GNN runtime must fail, never pass. The
  manuscript step asserts `rendered_documents is True`.
- New console entry points belong in the `tests` job's wheel smoke loop as well as in
  `pyproject.toml`.
- Keep the `tests` job's replay capture path in step with
  `docs/okn/evidence/2026-10-23-entity-model`.
- Parent rules: [../../AGENTS.md](../../AGENTS.md).
