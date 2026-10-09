---
title: API documentation instructions
type: agents
status: stable
updated: 2026-10-09
---

# API documentation instructions

Document only symbols that can be imported from the installed `cognitive` or
`Things` packages. Every Python example must compile and run, and use a real
public constructor; run it with `.venv/bin/python` before keeping it. Keep
import paths explicit, include configuration requirements, and update
`docs/api/api_reference.md` when an exported symbol or console script changes.
Pages for areas with no importable code (`ant_colony_api.md`,
`research_api.md`) say so rather than describing planned interfaces.

Run `python code/scripts/validate_docs.py --json` after editing API prose.
