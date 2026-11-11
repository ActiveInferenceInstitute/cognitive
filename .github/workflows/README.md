---
title: CI workflows
type: documentation
status: stable
updated: 2026-11-03
---

# `.github/workflows/`

[`quality.yml`](quality.yml) runs on every push and pull request. Its jobs:

| Job | What it proves |
|---|---|
| `tests` | The suite on Python 3.12 with branch coverage over the configured source trees. |
| `checks` | Ruff, mypy, compileall, the documentation and link gate, OKN submission diagnostics, semantic-review freshness, the manuscript pipeline and the benchmarks. |
| `gnn conformance (pinned)` | `specs/discrete_variational.gnn.md` against the parser commit in `specs/gnn-pin.json`. |
| `okn-replay` | Replays captured evidence and checks the separate join explanations. |
| `mutation-gate` | Every declared mutation is detected by the suite. |
| `build-package` | Wheel build, `twine check`, install, and every console entry point. |

`concurrency` cancels a superseded run on the same ref.
