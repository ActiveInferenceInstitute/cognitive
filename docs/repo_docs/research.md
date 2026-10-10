---
title: Research workflow
type: guide
status: stable
created: 2024-02-12
updated: 2026-10-10
tags:
  - research
  - guide
  - methodology
semantic_relations:
  - type: implements
    links:
      - "[[docs/repo_docs/documentation_standards|Documentation standards]]"
  - type: relates
    links:
      - "[[docs/research/README|Research documentation]]"
      - "[[docs/repo_docs/validation|Validation]]"
---

# Research workflow

Where research material lives in this repository and how to keep a result
reproducible. This page does not prescribe a research methodology; it names the
tools that exist.

## Where things are

| Material | Location |
|---|---|
| Research notes, plans, architectures | `docs/research/` ([[docs/research/README|index]]) |
| Research topics as vault pages | `knowledge_base/research/` |
| Cited works, one note each | `knowledge_base/citations/` |
| The executable manuscript | `docs/manuscript/` ([[docs/manuscript/README|README]]) |
| Runnable agents | `code/Things/Simple_POMDP`, `code/Things/Continuous_Generic` |
| Timing benchmarks | `cognitive-benchmark` |

## Keeping a result reproducible

- **Seed everything.** `InferenceConfig(seed=...)`, the `seed` field in the
  `SimplePOMDP` configuration, and `ContinuousActiveInference(seed=...)` make
  runs repeatable.
- **Generate numbers; do not type them.** The manuscript's numeric values and
  figures are produced by `cognitive-build-manuscript`, which hydrates the
  `{{...}}` variables in `docs/manuscript/` from the package. A number in a
  section that is not a variable is a defect.
- **Cite by key.** Citations in the manuscript are BibTeX keys in
  `docs/manuscript/references.bib`; `validate_docs.py` fails if a key is missing.
- **Keep negative results.** A result that did not hold is recorded as a result.
- **Name the limits.** State which agents are executable and which folders are
  design notes only (the folders under `code/Things/` other than the two above).

## Running an experiment

```bash
python -m pytest -q                                  # the behaviour checks
cognitive-benchmark --repetitions 10 --output benchmarks.json
cognitive-build-manuscript --output build/manuscript --no-pdf
```

`--no-pdf` skips LaTeX; `--no-render` additionally skips Pandoc. The build
directory is git-ignored.

## Writing it up

Research pages follow [[docs/repo_docs/documentation_standards|Documentation standards]]
and the templates in `docs/templates/` (`research_document.md`,
`experiment_template.md`, `analysis_template.md`). Fill every field, cite
sources as `knowledge_base/citations/` links, and run
`python code/scripts/validate_docs.py --json` before committing.

## Related documentation

- [[docs/research/README|Research documentation]]
- [[docs/manuscript/README|Executable manuscript]]
- [[docs/repo_docs/simulation|Simulation]]
- [[docs/repo_docs/machine_learning|Machine learning]]
