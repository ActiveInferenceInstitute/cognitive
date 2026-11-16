---
title: OKN evidence captures
type: documentation
status: stable
updated: 2026-11-03
---

# `docs/okn/evidence/`

One directory per captured cross-graph run, named by date (and topic where
there were two on a day): `2026-10-13`, `2026-10-20`,
`2026-10-20-graph-methods`, `2026-10-23-entity-model` and `2026-11-16-current-corpus`. Each holds a
`manifest.json` recording the endpoint, observation time, query and response
SHA-256 digests and the local graph digest, plus the captured SPARQL responses
and the replay query.

The `tests` CI job's replay step replays `2026-11-16-current-corpus`:
`daf-okn join --replay docs/okn/evidence/2026-11-16-current-corpus`. The parent
[`../README.md`](../README.md) describes the export, and
[`../AGENTS.md`](../AGENTS.md) states what is generated.
