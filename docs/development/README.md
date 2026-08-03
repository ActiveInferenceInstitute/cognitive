---
title: Development resources
type: documentation
status: stable
---

# Development resources

The project is a Python package configured by the repository root
`pyproject.toml`. Use an editable install with the development extra:

```bash
python -m pip install -e ".[dev]"
```

The supported development loop is:

```bash
python -m pytest -q
ruff format code/Things code/tools/src code/scripts code/tests
ruff check .
mypy code/tools/src code/Things code/scripts
python -m compileall -q code
python code/scripts/validate_docs.py --json
python code/scripts/verify_links.py . --json
python code/scripts/check_markdown_links.py . --json
```

`validate_docs.py` already includes the standard Markdown link check; the
standalone `check_markdown_links.py` script also reports heading-anchor
warnings (renderer-dependent) that the gate treats as advisory.

Use temporary directories for figures, animations, benchmark JSON, and
manuscript renders. The CI workflow in `.github/workflows/quality.yml` runs
the same gates. The full manuscript workflow is documented in
[`docs/manuscript/README.md`](../manuscript/README.md).
