---
title: Mutation testing
type: tool
status: stable
created: 2026-08-19
tags:
  - testing
  - mutation
  - tools
semantic_relations:
  - type: documents
    links:
      - '[[coverage_tools|Coverage tools]]'
      - '[[../development/README|Development resources]]'
---

# Mutation testing

Coverage reports that a line executed. It does not report whether a test would
notice if that line were wrong. Both facts matter, and only the second one is
about correctness.

```bash
python -m scripts.mutation_check          # human-readable
python -m scripts.mutation_check --json   # machine-readable
python -m scripts.mutation_check --filter gradient
python -m pytest --mutation code/tests/test_mutation_check.py
```

## How it works

`code/scripts/mutation_check.py` holds a list of `Mutation` records. Each is a
precise `(file, original, mutated)` source edit with a rationale and the tests
that should reject it. For every entry the harness writes the mutated source to
disk, runs those tests in a subprocess, restores the file, and records whether
the mutation was **killed**.

A mutation that the suite still passes is a **survivor**: the line is executed
by the tests but not checked by them. Every survivor is either a missing
assertion or a mutation that does not describe a real defect.

## Two details that decide whether the result means anything

**Bytecode cache isolation.** Python decides whether a cached `.pyc` is stale by
comparing the source's modification time and size. Several mutations here
replace a character with another of the same width — `limit < 1` becomes
`limit < 0`. Written inside the same clock tick, such an edit changes neither
field, the interpreter loads the previous bytecode, and the mutation never
reaches the tests, which then pass and are reported as a survivor. Each run
therefore points `PYTHONPYCACHEPREFIX` at a fresh directory. This is not
hypothetical: it produced an intermittent false survivor before it was fixed.

**A broken import is not a kill.** A non-zero pytest exit only counts when a
test actually failed. A mutation that breaks collection proves the file no
longer parses, not that a wrong answer would be caught. The harness raises on
that case rather than scoring it.

## What belongs in the list

Every entry corresponds to a defect this package shipped, or to the boundary a
defect turned on: the expected-free-energy double count, the helper's wrong
ambiguity term, the missing dynamics Jacobian transpose, the entropy axis
mismatch, the homeostatic preference mapping, the belief-blend learning rate,
the policy temperature, the policy aggregation, and the precision clip.

When a bug is fixed, add the mutation that reproduces it. That keeps the test
which caught it load-bearing, so a later refactor cannot quietly remove the
assertion and leave a green suite behind.

`code/tests/test_mutation_check.py` proves the harness can report a survivor,
restores every file it edits, and refuses to run against a source whose anchor
has moved. A mutation gate that cannot fail would print a perfect score
whatever the suite did.
