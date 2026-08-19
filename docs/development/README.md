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
python -m pytest -q --cov
ruff format code/Things code/tools/src code/scripts code/tests
ruff check .
mypy code/tools/src code/Things code/scripts
python -m compileall -q code
python code/scripts/validate_docs.py --json
python code/scripts/verify_links.py . --json
python code/scripts/check_markdown_links.py . --json
python -m scripts.build_manuscript --no-render --output "${TMPDIR:-/tmp}/manuscript"
cognitive-benchmark --repetitions 1 --output "${TMPDIR:-/tmp}/benchmarks.json"
python -m scripts.mutation_check
```

`mutation_check` applies a declared set of source mutations and requires the
suite to reject each one. It answers the question coverage cannot: whether a
covered line is also a checked line. See
[`../tools/mutation_testing.md`](../tools/mutation_testing.md). It spawns one
pytest subprocess per mutation, so run it before opening a change rather than on
every save.

`--cov` measures branch coverage across every module in the configured source
tree against the floor in `pyproject.toml`; see
[`../tools/coverage_tools.md`](../tools/coverage_tools.md) for why the report
carries no include filter.

The last two commands exercise the installed entry points. `--no-render` runs
every stage the package owns and skips only Pandoc and XeLaTeX, so the
manuscript pipeline stays checkable without a document toolchain. Drop the flag
to render HTML and the PDF.

`validate_docs.py` already includes the standard Markdown link check; the
standalone `check_markdown_links.py` script also reports heading-anchor
mismatches for standard links, and `verify_links.py` reports
`anchor_warnings` for wiki links whose `#fragment` does not match a
heading slug. Both are advisory (informational; they do not fail the
gate).

The documentation term policy those gates enforce lives in
[`../policy/documentation_terms.yaml`](../policy/documentation_terms.yaml)
rather than inside the validator; see
[`../policy/README.md`](../policy/README.md).

Use temporary directories for figures, animations, benchmark JSON, and
manuscript renders. The CI workflow in `.github/workflows/quality.yml` runs
the same gates, and additionally builds the distribution, installs the wheel,
and runs every declared console script. The full manuscript workflow is documented in
[`docs/manuscript/README.md`](../manuscript/README.md).
