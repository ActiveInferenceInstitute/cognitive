---
title: Development resources
type: documentation
status: stable
updated: 2026-11-12
---

# Development resources

The project is configured by the repository root `pyproject.toml` and `uv.lock`.
Use the shared root entry point for local work and CI. It invokes
`uv run --locked --extra dev --python 3.12 python code/scripts/gate.py ARM`;
`uv` prepares the locked development environment without a separate editable
installation step.

## Shared gates

Run from the repository root:

```bash
./run.sh                 # defaults to test
./run.sh lint
./run.sh types
./run.sh test
./run.sh test-full
./run.sh docs
./run.sh gnn
./run.sh manuscript
```

| Arm | Scope |
|---|---|
| `lint` | Ruff lint, Ruff format check, and compileall. It checks formatting without rewriting files. |
| `types` | mypy over the configured package, agents, and repository scripts. |
| `test` | Selected core-model test files listed below, with `-m 'not visualization'`, without coverage. |
| `test-full` | The entire suite, including visualization, with branch coverage over the configured source tree and its configured floor. |
| `docs` | `validate_docs.py`, documentation-index freshness, OKN submission and quality checks, and semantic-review freshness. |
| `gnn` | The GNN validator and conformance tests using the separately provisioned pinned parser interpreter. |
| `manuscript` | The no-render pipeline followed by HTML rendering; add `--pdf` to include PDF rendering. |

The default `test` arm provides narrower core-model feedback, using only:

- `code/tests/test_active_inference_api.py`
- `code/tests/test_dispatcher_semantics.py`
- `code/tests/test_simple_pomdp.py`
- `code/tests/test_math_oracles.py`

It also applies `-m 'not visualization'` and does not collect coverage.
This selection is not equivalent to the full suite's coverage or workload and
does not replace `test-full`. Coverage scope for `test-full` is defined in
`pyproject.toml`; do not narrow it with a `--cov=<path>` override.
See [coverage tools](../tools/coverage_tools.md).

The before fast-test sample runs this same file and marker selection against
the original source; the repository previously had no named fast arm. The
generated report retains command selections and exit codes. Single samples
do not establish a universal speedup.

The docs arm runs `docs_index.py --check` and `gate_evidence.py --check` to
check generated navigation and measured-evidence freshness without rewriting
them.

### Provisioned tools

The GNN arm requires `GNN_RUNTIME_PYTHON` to name the pinned parser's Python
interpreter. Provisioning is an explicit network operation:

```bash
./run.sh gnn --provision
GNN_RUNTIME_PYTHON=/path/to/gnn-runtime/bin/python ./run.sh gnn
```

The pin and parser-integrity contract are documented in
[Model specification](../gnn.md). An unavailable parser is not a conformance pass.

The manuscript arm needs Pandoc and pandoc-crossref for HTML; `--pdf` also
requires XeLaTeX. For a no-render-only invocation and renderer setup, see the
[manuscript guide](../manuscript/README.md).

## Documentation maintenance

`validate_docs.py` checks frontmatter, Python fences, public exports, term
policy, wiki and Markdown links, knowledge-base catalogs, manuscript references,
and committed RDF documentation. The root docs arm adds the OKN submission,
quality, and semantic-review checks. The term policy lives in
[`../policy/documentation_terms.yaml`](../policy/documentation_terms.yaml).
Heading-anchor mismatches reported by the link tools remain advisory.

The [documentation hub](../README.md) is curated navigation; the separate
[documentation index](../documentation-index.md) is generated recursively from
long-lived Markdown under `docs/`, excluding agent instructions and the index
itself. It includes the generated gate-measurement page. Refresh the appropriate
catalogs after adding or renaming a page, then run the shared docs gate:

```bash
uv run --locked --extra dev --python 3.12 python code/scripts/docs_index.py --write .
uv run --locked --extra dev --python 3.12 python code/scripts/kb_index.py --write .
./run.sh docs
```

Use `docs_index.py --check .` to check the flat index without writing it.
The knowledge-base catalogs and the documentation index cover different trees.
Generated gate measurements are recorded in
[`gate-measurements.md`](gate-measurements.md), from
[`gate-measurements.json`](gate-measurements.json); keep measured counts and
execution timings there rather than copying them into guides.

For a future comparison, preserve a before record before editing, then record
the after phase serially on the same host. Each arm accepts `--record FILE`;
redirect its output to a sibling `PHASE-ARM.log` to retain test and coverage
observations. Provision dependencies and the pinned parser before timing.
Do not compare different selections as equivalent work or use failed arms
to claim a speedup.

```bash
mkdir -p output/gate-measurements
BASELINE_REVISION=$(git rev-parse HEAD)
./run.sh test --record output/gate-measurements/before.json > output/gate-measurements/before-test.log 2>&1
# After editing and recording every after arm:
uv run --locked --extra dev python code/scripts/gate_evidence.py --refresh \
  --before output/gate-measurements/before.json \
  --after output/gate-measurements/after.json \
  --baseline-revision "$BASELINE_REVISION"
uv run --locked --extra dev python code/scripts/gate_evidence.py --check
```

The refresh script derives the Markdown from the preserved JSON and logs and
binds it to the gate sources. A source change makes the checked evidence stale;
keep the original records instead of editing measured values.


## Additional checks and CI

The benchmark and mutation checks remain available independently:

```bash
uv run --locked --extra dev --python 3.12 cognitive-benchmark --repetitions 1 --output "${TMPDIR:-/tmp}/benchmarks.json"
uv run --locked --extra dev --python 3.12 python -m scripts.mutation_check
```

The mutation check changes source temporarily and must run alone after other
tests finish. It checks that the suite rejects each declared source mutation;
see [mutation testing](../tools/mutation_testing.md).

Build and package-validation tools are locked development dependencies. The
captured OKN replay uses the optional `replay` extra, including pinned
`daf-okn` and `rdflib==7.6.0`, through
`uv run --locked --extra dev --extra replay` only for replay steps.

Use temporary directories for figures, animations, benchmark JSON, and
manuscript renders. The [quality workflow](../../.github/workflows/quality.yml)
defines a consolidated `tests` job and an independent `gnn-conformance` job
(display name `gnn conformance (pinned)`). Both use the locked project environment
and the shared setup action; the parser keeps its separate interpreter.
See [CI tools](../tools/ci_tools.md) for the job contract.

Hosted Actions execution is currently blocked by the organization's billing
lock. Workflow configuration and local measurements are not evidence of a
successful hosted run.
