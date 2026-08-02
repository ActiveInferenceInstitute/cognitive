---
title: Getting Started
type: guide
status: stable
semantic_relations:
  - type: supports
    links:
      - '[[README|Repository README]]'
      - '[[docs/development/README|Development resources]]'
      - '[[docs/examples/README|Executable examples]]'
      - '[[docs/manuscript/README|Executable manuscript]]'
---

# Getting Started

This repository contains a Python package for discrete and continuous Active
Inference models, supporting utilities, examples, tests, and an Obsidian-style
knowledge base. The root README is the shortest working introduction; this page
adds the setup and verification details needed for a fresh checkout.

## Prerequisites

- Python 3.10 or newer (the floor declared in `pyproject.toml`).
- A virtual environment is recommended.
- A LaTeX/XeLaTeX installation is optional and is only required for a PDF
  manuscript build.

## Install

From the repository root:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

On Windows, use `.venv\\Scripts\\python` instead of `.venv/bin/python`.

The editable install provides the `cognitive`, `Things`, and `scripts` packages
and these console commands:

```bash
cognitive-create-node --help
cognitive-verify-links . --json
cognitive-validate-docs . --json
cognitive-benchmark --repetitions 1
```

## Run a discrete model

The root [`README.md`](../../README.md) contains a complete two-state example
using `DiscreteGenerativeModel`, `InferenceConfig`, `ModelState`, and
`ActiveInferenceDispatcher`. It is the recommended first runtime example because
all of those symbols are public package exports.

The conceptual knowledge base is background material, not a substitute for the
runtime API. For implementation behavior, use the package source and tests.

## Verify the checkout

Run the same checks used by the repository's quality workflow:

```bash
python -m pytest -q
ruff check .
mypy code/tools/src code/Things code/scripts
python -m compileall -q code
python code/scripts/validate_docs.py --json
python code/scripts/verify_links.py . --json
```

The documentation validator checks YAML frontmatter, Python fenced examples,
public exports, manuscript references, and explicit wiki links. Unresolved
extensionless wiki targets may be intentional concept links; use
`--strict-wiki-links` when auditing those targets as files.

## Build the manuscript

After installation:

```bash
cognitive-build-manuscript --output build/manuscript
```

Use `--no-pdf` if XeLaTeX is unavailable. The build writes HTML, figures,
provenance manifests, and (when enabled) PDF output to the requested directory;
these generated files are ignored by Git. See
[`docs/manuscript/README.md`](../../docs/manuscript/README.md) for the source
section map and build contract.

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

Code and executable package material are distributed under the MIT License in
[`LICENSE`](../../LICENSE). Documentation and knowledge-base content are marked
CC BY-NC-SA 4.0 in the repository's documentation notices.
