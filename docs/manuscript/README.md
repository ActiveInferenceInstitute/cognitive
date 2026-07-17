# Cognitive Active Inference manuscript

This directory is the publication source for the Cognitive Active Inference
repository. It follows the numbered-section, configuration, bibliography,
figure-registry, and cross-reference conventions used by the
`template_code_project` exemplar in the Institute template repository.

The manuscript describes only the executable package in this repository. Its
numeric values and figures are hydrated by `code/scripts/build_manuscript.py`
from `config.yaml` and the public `cognitive` and `Things` APIs.

## Build

From the repository root, after installing the package:

```bash
python -m pip install -e ".[dev]"
cognitive-build-manuscript --output build/manuscript
```

The build creates `combined.md`, `manuscript.html`, `manuscript.pdf`,
`figures/`, `figure_registry.json`, `manuscript_variables.json`, and
`build_manifest.json` under the requested output directory. Build output is
ignored by Git. Use `--no-pdf` when a LaTeX installation is unavailable.

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
