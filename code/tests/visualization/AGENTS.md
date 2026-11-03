---
title: Visualization test instructions
type: agents
status: stable
updated: 2026-11-03
---

# AGENTS.md — `code/tests/visualization/`

- Keep generated figures under `tmp_path`. A test that writes into the checkout
  leaves artifacts in a tracked directory.
- Mark a figure-producing test with `@pytest.mark.visualization`;
  `--strict-markers` rejects an unregistered marker.
- Assert on the produced file or figure, not only that the call returned.
- Parent rules: [../AGENTS.md](../AGENTS.md).
