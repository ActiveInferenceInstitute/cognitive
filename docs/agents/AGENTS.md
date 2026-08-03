---
title: Agent documentation instructions
type: agents
status: stable
created: 2024-01-01
updated: 2026-08-02
tags:
  - agents
  - documentation
  - clearinghouse
  - cognitive
semantic_relations:
  - type: organizes
    links:
      - '[[README|Agent documentation index]]'
      - '[[agent_docs_readme|Agent documentation clearinghouse]]'
      - '[[../repo_docs/getting_started|Getting started]]'
---

# Agent documentation instructions

The `agents/` folder documents agent architectures conceptually. Executable
agent behavior lives in `code/Things/` and in the package under
`code/tools/src/models/active_inference/`; theoretical foundations live in
`knowledge_base/agents/`.

## Sources of truth

- `code/Things/AGENTS.md` — index of the self-contained agent implementations.
- `code/tools/src/models/active_inference/AGENTS.md` — the validated discrete
  Active Inference model family.
- `knowledge_base/agents/README.md` — conceptual agent architecture material.
- `docs/examples/README.md` — executable example commands.

## Rules

- Describe real implementations by their actual import paths and constructors.
- Conceptual agent write-ups must not define classes that do not exist in the
  package, and must not be presented as working examples.
- Keep the clearinghouse (`agent_docs_readme.md`) as a navigation aid; every
  entry it names must exist.
