# Reproducibility Certification {#sec:reproducibility}

Reproducibility is treated as a chain of executable checks rather than a
statement about intention. A clean build performs the following sequence:

```text
config.yaml experiment declaration
    -> validated DiscreteGenerativeModel
    -> seeded dispatcher and continuous agent
    -> computed variable mapping (recorded in manuscript_variables.json)
    -> hydrated numbered sections and generated figures
    -> validated figure registry and combined-source hash
    -> Pandoc-crossref HTML and, when requested, XeLaTeX PDF
```

The computed variable mapping is shared by the generated JSON and hydrated
prose; numerical results are not maintained independently in the manuscript.
The builder writes a `build_manifest.json` containing the package version,
configuration path, hydrated variables, figure registry, and SHA-256 hash of
`combined.md`. Before rendering, every registered figure must be a regular
nonempty file, and registry labels and paths must agree with the manuscript.
The formal source-to-state contract is:

$$
\operatorname{load}(\operatorname{save}(S)) = S
$$ {#eq:state_round_trip}

for every validated state $S$. YAML persistence in
`code/tools/src/models/active_inference/base.py` writes a version field,
finite arrays, and a temporary file before replacing the destination. The
regression suite checks a discrete round trip and the compact POMDP round
trip.

## Required gates

From the repository root:

```bash
./run.sh lint
./run.sh types
./run.sh test
./run.sh test-full
./run.sh docs
./run.sh gnn
./run.sh manuscript
./run.sh manuscript --pdf
```

The root runner invokes `uv run --locked --extra dev --python 3.12 python
code/scripts/gate.py` with the selected arm; a bare `./run.sh` selects `test`.
`lint` checks Ruff lint and formatting plus compilation; `types` runs mypy.
`test` runs selected core-model tests excluding visualization, without coverage,
from the existing `active_inference_api`, `dispatcher_semantics`, `simple_pomdp`,
and `math_oracles` test modules. `test-full` runs the whole suite with branch
coverage. `docs` runs
`validate_docs` and the OKN submission, quality, and semantic freshness checks.
`gnn` runs the validator and conformance tests with a provisioned
`GNN_RUNTIME_PYTHON`; `./run.sh gnn --provision` explicitly allows network
provisioning of that independent parser interpreter.

`manuscript` first exercises the no-render pipeline and then renders HTML.
Its `--pdf` option additionally builds the full PDF. The authoritative
`code/scripts/build_manuscript.py` builder produces the figures, computed
variables, validated registry, hydrated prose, and publication artifacts;
the gate orchestrates it rather than introducing a separate build path.

## Provenance boundary

The repository tracks source, configuration, tests, bibliography, and build
scripts. Generated PDFs, HTML, figures, coverage data, and benchmark reports
are reproducible outputs and belong in an ignored build location. This keeps
the published source reviewable while preserving a deterministic path to the
rendered artifact.
