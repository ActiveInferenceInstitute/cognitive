---
title: Getting Started
type: guide
status: stable
updated: 2026-10-10
semantic_relations:
  - type: supports
    links:
      - "[[README|Repository README]]"
      - "[[docs/development/README|Development resources]]"
      - "[[docs/examples/README|Executable examples]]"
      - "[[docs/manuscript/README|Executable manuscript]]"
---

# Getting Started

This repository contains a Python package for discrete and continuous Active
Inference models, supporting utilities, examples, tests, and an Obsidian-style
knowledge base. The root README is the shortest working introduction; this page
adds the setup and verification details needed for a fresh checkout.

## Prerequisites

- The package requires Python 3.11 or newer. Development gates use Python 3.12
  and the dependencies in `uv.lock`.
- Install `uv` and use a shell with Bash to run the development gates.
- Manuscript HTML rendering needs Pandoc and pandoc-crossref. PDF rendering also
  needs XeLaTeX.

## Install

From the repository root, prepare the locked development environment and check
that the command is available.

```bash
uv run --locked --extra dev --python 3.12 cognitive --help
```

The environment includes the `cognitive`, `Things`, and `scripts` packages.
See the [[docs/tools/README|tools guide]] for the knowledge-base authoring commands.

## Run a discrete model

The root [`README.md`](../../README.md) contains a complete two-state example
using `DiscreteGenerativeModel`, `InferenceConfig`, `ModelState`, and
`ActiveInferenceDispatcher`. It is the recommended first runtime example because
all of those symbols are public package exports.

The conceptual knowledge base is background material, not a substitute for the
runtime API. For implementation behavior, use the package source and tests.

## Verify the checkout

Run the core-model tests and documentation checks from the repository root.

```bash
./run.sh test
./run.sh docs
```

Use `./run.sh test-full` for the full suite and its branch coverage threshold.
The docs arm checks frontmatter, internal links, generated catalogs, manuscript
references, and OKN export quality and freshness. The
[[docs/development/README|development guide]] lists the additional gate arms
and the prerequisites for each.

## Build the manuscript

```bash
./run.sh manuscript
```

This runs the no-render pipeline and builds HTML under `build/manuscript`.
Use `./run.sh manuscript --pdf` when XeLaTeX is available. See the
[[docs/manuscript/README|manuscript guide]] for the renderer setup, output
locations, and a no-render-only command.

## Where to go next

- [`docs/api/README.md`](../../docs/api/README.md): API documentation policy and
  reference entry points.
- [`docs/examples/README.md`](../../docs/examples/README.md): executable example
  commands.
- [`docs/development/README.md`](../../docs/development/README.md): development
  loop and CI gates.
- [`knowledge_base/README.md`](../../knowledge_base/README.md): conceptual and
  research navigation.
- [`docs/config/README.md`](../../docs/config/README.md): configuration files and
  current defaults.

## Licensing

Code is MIT-licensed ([`LICENSE`](../../LICENSE)). Repository-owned knowledge-base
content and graph exports are CC BY 4.0 ([license notice](../../LICENSE-KNOWLEDGE-BASE.md)).
Other documentation remains CC BY-NC-SA 4.0.
