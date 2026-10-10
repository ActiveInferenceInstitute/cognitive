---
title: Package documentation
type: guide
status: stable
updated: 2026-10-10
tags:
  - documentation
  - api
  - docstrings
semantic_relations:
  - type: implements
    links:
      - "[[docs/repo_docs/documentation_standards|Documentation standards]]"
  - type: relates
    links:
      - "[[docs/api/api_reference|API reference]]"
      - "[[docs/repo_docs/api_development|API development]]"
---

# Package documentation

How the `cognitive` package and the `Things` agents are documented. Everything is
hand-written Markdown plus docstrings; there is no Sphinx, MkDocs, or generated
API site, and no tool that builds documentation from docstrings.

## Layers

1. **Docstrings** in `code/tools/src`, `code/Things`, and `code/scripts`. Public
   functions and classes should carry type hints and a docstring. The usual style is
   a short summary line, then prose that states behavior, units, and what raises,
   with identifiers in double backticks. Args/Returns sections are rare; this
   abridged excerpt from `DiscreteGenerativeModel.posterior` shows the shape:

   ```text
   def posterior(self, observation: int | np.ndarray, prior: np.ndarray) -> np.ndarray:
       """Return ``P(s | o)`` for one observation index or an observation distribution.

       Fractional indices are rejected rather than truncated.
       """
   ```

2. **Folder pages.** Each package folder has a `README.md` and usually an
   `AGENTS.md` (for example `code/tools/src/models/active_inference/`). They state
   what the folder owns and its invariants.
3. **API pages** in `docs/api/`. [[docs/api/api_reference|The API reference]] lists
   the exports of `cognitive.__all__` and the agent classes with their signatures;
   [[docs/api/api_examples|API examples]] holds runnable snippets. The root
   `README.md` carries the minimal discrete example.
4. **Guides and examples** in `docs/guides/`, `docs/examples/`, and
   `docs/implementation/`, which link to the API pages instead of repeating them.
5. **Tool notes** in `docs/tools/`.

## Rules

- Name a class or function in documentation only if it can be imported, and give
  the import path.
- A `python` fence that uses the package must run as written. The validator
  checks only that it compiles, so run changed snippets yourself.
- When a public symbol changes, update its docstring, the API page, and
  `CHANGELOG.md` in the same commit.
- Keep folder pages and the code in agreement. `validate_docs.py` verifies that
  every name in `cognitive.__all__` exists; it does not verify prose.

## Checking

```bash
python code/scripts/validate_docs.py --json
python -c "import cognitive; print(sorted(cognitive.__all__))"
```

Type checking (`mypy code/tools/src code/Things code/scripts`) and `ruff check .`
cover the signatures the docstrings describe. See
[[docs/repo_docs/validation|Validation]].

## Related documentation

- [[docs/repo_docs/documentation_standards|Documentation standards]]
- [[docs/repo_docs/api_development|API development]]
- [[docs/api/api_reference|API reference]]
