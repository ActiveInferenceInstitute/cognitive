---
title: OKN evidence instructions
type: agents
status: stable
updated: 2026-11-03
---

# AGENTS.md — `docs/okn/evidence/`

- These are captured responses, not authored pages. Do not hand-edit a `.json`,
  `.rq` or `manifest.json`.
- A new capture is a new dated directory written by `daf-okn join`; it records
  the endpoint, the observed time and the digests in its manifest.
- `2026-10-23-entity-model` is replayed by CI. Keep the path in
  `.github/workflows/quality.yml` and here in step.
- Do not add files to a capture directory; replay and explanation checks bind
  the directory's contents.
- Parent rules: [../AGENTS.md](../AGENTS.md).
