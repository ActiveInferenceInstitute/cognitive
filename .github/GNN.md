# GNN conformance

The `gnn-conformance` job in `.github/workflows/quality.yml` has the display
name **`gnn conformance (pinned)`**. Preserve both identifiers for check
consumers. It is independent of the consolidated `tests` job because the
upstream parser needs its own interpreter.

The single canonical importer pin is [`specs/gnn-pin.json`](../specs/gnn-pin.json),
with schema **`gnn-importer-pin/v1`**, repository
`https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation`,
tag **`v5.0.3`**, and commit
**`8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`**. It also records the SHA-256
of each parser file. The job checks that the tag still peels to the pinned
commit, provisions that commit, and verifies parser digests before trusting
its verdict. The JSON is the declaration consumed by tooling; this page is a
pointer, not another machine-readable pin.

Only [`specs/discrete_variational.gnn.md`](../specs/discrete_variational.gnn.md),
the `Cognitive Discrete Variational Model`, is parsed. The job does not scan
for other `*.gnn.md` files. `./run.sh gnn`, with `GNN_RUNTIME_PYTHON` set,
runs the validator and conformance tests; `./run.sh gnn --provision` is the
explicit network-enabled provisioning path.

The validator compares declared `A`, `B`, `C`, `D`, and `E`, dimensions, and
horizon with the canonical configuration
`code/tools/src/models/active_inference/config_examples/discrete_variational.yaml`
as loaded by `ActiveInferenceFactory.create_from_yaml`. It also compares the
pinned parser's reading with the repository's reader. A missing parser runtime
is not a pass. See [Model specification](../docs/gnn.md).

The job uses
`ActiveInferenceInstitute/.github/.github/actions/setup-python-uv@7159e67315330aac8e414aebae0ad1e999a41448`
and `uv.lock` for the project environment; the parser interpreter is separate.
Hosted Actions execution is currently blocked by the organization's billing
lock, so the configured job is not evidence of hosted success.
