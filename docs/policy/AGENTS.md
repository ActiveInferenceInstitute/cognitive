---
title: Repository Policy Instructions
type: instructions
status: stable
updated: 2026-10-10
---

# Working with repository policy files

## What lives here

Machine-readable policy consumed by the repository's validation gates. Nothing
here is documentation prose that a gate ignores; every file is loaded and
enforced at validation time.

## Rules for changing a policy file

- A policy file is loaded by a named script. Find that script before editing —
  `documentation_terms.yaml` is read by `load_term_policy` in
  `code/scripts/validate_docs.py` — and confirm the shape it expects.
- Changing a policy changes what the gate accepts. Run
  `python code/scripts/validate_docs.py --json` after any edit and read the
  `term_policy` block of the report, which records the file path, the number of
  terms loaded, and the exempt paths actually applied.
- Add a path to `exempt_paths` only when the file's purpose is to discuss the
  policy. Exempting a file to make a failure go away hides the failure.
- Removing a term is an editorial decision about what the repository is willing
  to claim, not a cleanup. State the reason in the commit message.

## Adding a new policy file

Add the file, the loader that reads it, a regression test that proves an
invalid policy is rejected, and a row in [`README.md`](README.md). A policy
file with no loader is inert and misleads anyone who finds it.
