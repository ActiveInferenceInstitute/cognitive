---
title: API versioning
type: guide
status: stable
updated: 2026-10-09
---

# API versioning

The package version is declared in the root `pyproject.toml` (`project.version`)
and mirrored as `cognitive.__version__` in `code/tools/src/__init__.py`. The
current version is `1.1.0`.

## What enforces it

- `code/tests/test_version_consistency.py` fails if `cognitive.__version__`
  differs from `pyproject.toml`, or if [`CHANGELOG.md`](../../CHANGELOG.md) has no
  `## [<version>]` heading for the current version.
- `code/scripts/validate_docs.py` fails if a name in `cognitive.__all__` is not
  an attribute of the package.
- `CHANGELOG.md` states that versions follow semantic versioning and that
  entries which change a computed result are marked **behaviour change**, with
  what moved and why. Version 1.1.0 carries one: the discrete expected free
  energy is now `risk + ambiguity` (see the changelog for the migration note).

## Upgrade procedure

1. Install the editable package from the new revision:
   `python -m pip install -e ".[dev]"`.
2. Read the `CHANGELOG.md` entries between the old and new version.
3. Run `python -m pytest -q` and the static checks listed in
   [`../development/README.md`](../development/README.md).
4. Rebuild the manuscript: `cognitive-build-manuscript --output build/manuscript`.
5. Inspect `build_manifest.json` in the output directory.
