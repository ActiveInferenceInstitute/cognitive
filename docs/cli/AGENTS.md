---
title: Command-line documentation instructions
description: Rules for editing the command-line documentation so that it matches the argparse definitions.
type: agents
status: stable
updated: 2026-10-13
semantic_relations:
  - type: documents
    links:
      - "[[README|Command line]]"
      - "[[reference|Command-line reference]]"
---

# Command-line documentation instructions

## Ground truth

- The parser is built in `code/tools/src/cli.py` (`build_parser`). The option
  list in [`reference.md`](reference.md) comes from `cognitive <command> --help`.
- `cognitive benchmark` forwards its arguments to `cognitive.benchmarks.main`;
  document its options from that program, not from the parser above.
- Output file names come from `_simulate_command` and `_sweep_command`; the
  report command writes what `scripts/kb_report.py` `main` writes.
- The console scripts are declared in `pyproject.toml` under
  `[project.scripts]`.

## Rules

1. When an option is added or renamed, update `reference.md` in the same
   change. `code/tests/test_docs_sync.py` fails when a subcommand option is
   missing from the page.
2. Show commands that run from a fresh checkout and state the files they write.
3. State exit status for every failure the command can report; do not describe
   behaviour the code does not have.
4. Link settings to [`../config/settings.md`](../config/settings.md) instead of
   repeating their defaults here.

## Verification

```bash
python -m pytest code/tests/test_docs_sync.py code/tests/test_cli.py -q
cognitive simulate --help
```
