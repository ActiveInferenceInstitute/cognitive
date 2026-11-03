---
title: OKN evidence captures
type: documentation
status: stable
updated: 2026-11-03
---

# `docs/okn/evidence/`

One directory per captured cross-graph run, named by date (and topic where
there were two on a day): `2026-10-13`, `2026-10-20`,
`2026-10-20-graph-methods` and `2026-10-23-entity-model`. Each holds a
`manifest.json` recording the endpoint, observation time, query and response
SHA-256 digests and the local graph digest, plus the captured SPARQL responses
and the replay query.

The `okn-replay` CI job replays `2026-10-23-entity-model`:
`daf-okn join --replay docs/okn/evidence/2026-10-23-entity-model`. The parent
[`../README.md`](../README.md) describes the export, and
[`../AGENTS.md`](../AGENTS.md) states what is generated.
