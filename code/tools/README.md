# Cognitive package tools

The installable package source lives in `code/tools/src` and is exposed as
`cognitive` by the repository's `pyproject.toml`.

The public runtime surface is intentionally small:

```python
from cognitive import (
    ActiveInferenceDispatcher,
    DiscreteGenerativeModel,
    HomeostaticFactory,
    InferenceConfig,
)
from cognitive.models.matrices.matrix_ops import MatrixOps
```

Install from the repository root with:

```bash
python -m pip install -e ".[dev]"
```

Use `python -m pytest -q` for the complete test suite. Runtime output paths
are supplied by callers and tests use temporary directories.
