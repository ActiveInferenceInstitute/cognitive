# GNN conformance

`gnn conformance (pinned)` is the `gnn-conformance` job in
`.github/workflows/quality.yml`. It reads `specs/gnn-pin.json`, which pins
`ActiveInferenceInstitute/Generalized_Notation_Notation` at `v5.0.3`
(`8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`). The job first checks that the tag
still peels to that commit, installs the parser from that commit into a
separate runtime, and only then uses its verdict.

The job validates the one declaration this repository commits,
`specs/discrete_variational.gnn.md` (the `Cognitive Discrete Variational
Model`), by running `python -m scripts.validate_gnn_spec --gnn-runtime
"$RUNNER_TEMP/gnn-runtime/bin/python" --json` and then re-running `python -m
pytest -q code/tests/test_gnn_conformance.py` with `GNN_RUNTIME_PYTHON` set,
so the comparison against the pinned parser is required, not skipped. The
validator compares the declared `A`, `B`, `C`, `D`, and `E` matrices, their
dimensions, and the horizon with the canonical configuration
`code/tools/src/models/active_inference/config_examples/`
`discrete_variational.yaml` as loaded by
`ActiveInferenceFactory.create_from_yaml`, and compares the pinned parser's
reading of the same document with the repository's own reader. It does not
scan the rest of the repository for `*.gnn.md`, so a declaration committed
elsewhere is outside this job's scope.
