---
title: API documentation instructions
type: agents
status: stable
---

# API documentation instructions

Document only symbols that can be imported from the installed `cognitive` or
`Things` packages. Every Python example must compile and use a real public
constructor. Keep import paths explicit, include configuration requirements,
and update `docs/api/api_reference.md` when an exported symbol changes.

Run `python code/scripts/validate_docs.py --json` after editing API prose.
