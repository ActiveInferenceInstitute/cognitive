---
title: Setup tools
type: tool
status: stable
created: 2026-08-02
tags:
  - setup
  - install
  - tools
semantic_relations:
  - type: documents
    links:
      - '[[../repo_docs/getting_started|Getting started]]'
---

# Setup tools

A fresh checkout requires Python 3.10+ and an editable install:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

The editable install exposes the `cognitive`, `Things`, and `scripts`
packages and the `cognitive-*` console commands. See
`docs/repo_docs/getting_started.md` for the full setup and verification
steps.
