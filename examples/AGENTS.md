---
title: Examples instructions
description: Rules for adding or changing the example scripts and their configuration files.
type: agents
status: stable
updated: 2026-10-13
semantic_relations:
  - type: documents
    links:
      - "[[README|Examples]]"
---

# Examples instructions

## Rules

1. Keep scripts thin: load settings, call the library, write results. A script
   that needs logic the package lacks means the logic belongs in the package.
2. Name scripts `NN_subject.py`. Every script and every file in `configs/` must
   be listed in [`README.md`](README.md); `code/tests/test_docs_sync.py` checks.
3. Write only under `output/` by default and take `--output` to change it.
4. Pass `environ={}` to `load_settings` so results do not depend on the machine.
5. Use real computation. No stubs, fake data or placeholder results.

## Verification

```bash
python -m pytest code/tests/test_examples.py code/tests/test_docs_sync.py -q
```

`test_examples.py` runs each script and checks the files it writes.
