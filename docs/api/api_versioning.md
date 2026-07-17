---
title: API versioning
type: guide
status: stable
---

# API versioning

The package version is declared once in the root `pyproject.toml` and exposed
as `cognitive.__version__`. The current version is `1.0.0`.

## Change rules

- A correction that preserves accepted inputs and output contracts increments
  the patch component.
- A new validated public capability increments the minor component.
- A change to matrix orientation, configuration schema, persistence schema, or
  exported names increments the major component.

Every change updates tests, runtime-facing documentation, and the manuscript
when the mathematical contract changes. The public export check in
`code/scripts/validate_docs.py` prevents documentation from naming a symbol
that is absent from the installed package.

## Upgrade procedure

1. Install the editable package from the new revision.
2. Run `python -m pytest -q` and the static quality gates.
3. Validate documented configuration examples.
4. Rebuild `cognitive-build-manuscript --output build/manuscript`.
5. Review the generated `build_manifest.json` and PDF.
