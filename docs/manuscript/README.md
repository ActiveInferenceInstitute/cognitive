# Cognitive Active Inference manuscript

This directory is the publication source for the Cognitive Active Inference
repository. It follows the numbered-section, configuration, bibliography,
figure-registry, and cross-reference conventions used by the
`template_code_project` exemplar in the Institute template repository.

The manuscript describes only the executable package in this repository.
`config.yaml` declares the experiment; `code/scripts/build_manuscript.py`
computes its results through the public `cognitive` and `Things` APIs. The
computed variables are recorded in `manuscript_variables.json` and hydrate
the prose, rather than maintaining numerical claims separately.

## Build

From the repository root:

```bash
./run.sh manuscript
./run.sh manuscript --pdf
```

The manuscript gate first runs the no-render pipeline, then builds HTML.
`--pdf` additionally requires the full XeLaTeX PDF build. Pandoc and
`pandoc-crossref` are required for HTML; XeLaTeX is also required for PDF.
The authoritative builder remains `code/scripts/build_manuscript.py`, also
available as `cognitive-build-manuscript`. To invoke it directly for a full
publication build:

```bash
uv run --locked --extra dev --python 3.12 python code/scripts/build_manuscript.py --output build/manuscript
```

The full build creates `combined.md`, `manuscript.html`, `manuscript.pdf`,
`figures/`, `figure_registry.json`, `manuscript_variables.json`, and
`build_manifest.json` under the requested output directory. Build output is
ignored by Git. The builder's `--no-pdf` produces HTML without PDF;
`--no-render` generates and validates figures, variables, and hydrated source
without invoking the document toolchain. Both paths validate the registry
against regular nonempty generated files and manuscript figure labels,
paths, and cross-references before any rendering.

The manuscript source is intentionally split so that the lexicographic order
is the publication order:

| Source | Role |
| --- | --- |
| `00_abstract.md` | Abstract and contribution statement |
| `01_introduction.md` | Problem, scope, and formal object |
| `02_methodology.md` | Inference, control, validation, and persistence |
| `03_results.md` | Deterministic runtime evidence and figures |
| `04_conclusion.md` | Findings and engineering implications |
| `05_experimental_setup.md` | Configuration and measurement protocol |
| `06_reproducibility.md` | Build, validation, and provenance contract |
| `07_scope_and_related_work.md` | Scholarship, boundaries, and positioning |
| `08_appendix_formalism.md` | Extended equations and API correspondence |
| `09_appendix_validation.md` | Regression matrix and acceptance evidence |
| `99_references.md` | Bibliography entry point |

`SYNTAX.md` records the labels, citation keys, tokens, and figure contract.
`layer_contract.yaml` records the allowed relationship between manuscript
claims and executable files.
