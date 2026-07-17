---
title: Contribution guide
type: guide
status: stable
---

# Contribution guide

## Setup

```bash
git clone https://github.com/ActiveInferenceInstitute/cognitive.git
cd cognitive
python -m pip install -e ".[dev]"
```

## Implementation requirements

Changes to runtime behavior should include a regression test, explicit input
validation, a documented public import path, and an update to the executable
manuscript when the mathematical contract changes. Random behavior must accept
an explicit seed. File outputs must be supplied by the caller or resolved from
the configuration file.

Tests should exercise real numerical paths and inspect actual outputs. Use
`tmp_path` for generated files. Do not leave reports, figures, or build trees
in the repository.

## Required checks

```bash
python -m pytest -q
ruff check .
mypy code/tools/src code/Things code/scripts
python -m compileall -q code
python code/scripts/validate_docs.py --json
python code/scripts/verify_links.py . --json
cognitive-build-manuscript --output build/manuscript
```

Keep commits focused and include the observed validation results in the pull
request description. The package is published from `main` after the quality
workflow passes.
