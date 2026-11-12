---
title: OKN join explanations
type: documentation
status: stable
updated: 2026-11-03
---

# `docs/okn/explanations/`

Separate assertion-path explanations for the captured evidence, one directory per
run: `2026-10-23-entity-model/` holds `join-support.md`,
`join-support.json` and `support-manifest.json`.

They are generated from the capture by
`daf-okn explain-join --capture docs/okn/evidence/2026-10-23-entity-model
--output docs/okn/explanations/2026-10-23-entity-model --check`, which the
replay step in the `tests` CI job runs. The parent
[`../AGENTS.md`](../AGENTS.md) lists the generated artifacts.
