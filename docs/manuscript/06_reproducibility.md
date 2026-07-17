# Reproducibility Certification {#sec:reproducibility}

Reproducibility is treated as a chain of executable checks rather than a
statement about intention. A clean build performs the following sequence:

```text
config.yaml
    -> validated DiscreteGenerativeModel
    -> seeded dispatcher and continuous agent
    -> figure generator and hydrated manuscript tokens
    -> figure registry and combined-source hash
    -> Pandoc-crossref HTML and XeLaTeX PDF
```

The builder writes a `build_manifest.json` containing the package version,
configuration path, hydrated variables, figure registry, and SHA-256 hash of
`combined.md`. The formal source-to-state contract is:

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
python -m pytest -q
ruff check .
mypy code/tools/src code/Things code/scripts
python -m compileall -q code
python code/scripts/validate_docs.py --json
python code/scripts/verify_links.py . --json
cognitive-build-manuscript --output build/manuscript
```

The first five commands protect runtime and static integrity. The two
documentation commands inspect tracked text and explicit path-like links.
The final command proves that figures, variables, references, and rendered
artifacts can be produced from a fresh output directory.

## Provenance boundary

The repository tracks source, configuration, tests, bibliography, and build
scripts. Generated PDFs, HTML, figures, coverage data, and benchmark reports
are reproducible outputs and belong in an ignored build location. This keeps
the published source reviewable while preserving a deterministic path to the
rendered artifact.
