---
title: Repository helper script instructions
type: agents
status: stable
updated: 2026-11-03
---

# AGENTS.md — `docs/repo_docs/repo_scripts/`

- These are manual helpers. The gate is `python code/scripts/validate_docs.py
  --json`; a helper here never substitutes for it.
- `ruff` does not lint this directory. Read a script before running it, and do
  not present one as an installed command.
- A helper that rewrites files must print what it changed; do not run one over
  the tree as part of a gate.
- Parent rules: [../AGENTS.md](../AGENTS.md).
