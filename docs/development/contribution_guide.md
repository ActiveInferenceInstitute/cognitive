---
title: Contribution guide
type: guide
status: stable
updated: 2026-10-09
---

# Contribution guide

## Setup

Follow the [[docs/repo_docs/getting_started|getting-started guide]] to open a
fresh checkout in the locked Python 3.12 development environment.

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

A maintainer reviews the current pull request diff and resolves outstanding
review findings before merging. Run the applicable gates from a clean checkout
of that exact head commit. Documentation-only changes require `./run.sh docs`.
Changes to code, tests, dependency pins or workflows require these gates.

```bash
./run.sh lint
./run.sh types
./run.sh test-full
./run.sh docs
```

`test-full` enforces the branch coverage floor in `pyproject.toml`. For faster
feedback, `./run.sh test` runs the selected core-model tests. The
[[docs/development/README|development guide]] describes the local arms and the
additional checks. Run the pinned GNN conformance gate for changes to the model
specification, its runtime contract or the parser pin. For changes to the graph
export or join query, also replay the capture and verify its explanation.

## Publish the local gate result

The required GitHub status `tests` reports the verified local result for the
exact pull request head. Keep the passing source-bound receipts with the full
commit SHA, commands and exit results. Recheck the live head before publishing.
A new commit needs its own verification. The hosted quality workflow runs only
when explicitly dispatched and reports separate informational checks.

For GitVault-managed checkout owners, publish the result from the vault root
using the mapped identity and the reviewed `github-status` plan. Set
`REVIEWED_SHA` to the full commit SHA from the passing receipts.

```bash
parameters=$(printf \
  '{"repository":"ActiveInferenceInstitute/cognitive","expected_commit":"%s","context":"tests","state":"success","description":"Cognitive local gate passed"}' \
  "$REVIEWED_SHA")
./_tools/vault plan github-status --parameters "$parameters"
./_tools/vault github-apply --plan "$PLAN_ID"
./_tools/vault github-apply --plan "$PLAN_ID" --apply --confirm 'REPORT 1 COMMIT STATUS'
```

Use the returned plan ID for `PLAN_ID`. Review its exact commit, context and
result before applying it. Verify the GitHub readback is `tests: success` on the
current candidate, then merge through the approved pull request path.

After merging, identify the exact live `main` commit and compare its tree SHA
with the verified head's tree SHA. Run `./run.sh docs` on the integrated commit.
Also run `./run.sh test` when code or workflow files changed. If the trees differ,
repeat any additional applicable gates on the integrated tree. Record the exact
integrated SHA and gate results before retiring the topic work.

## Knowledge base changes

Regenerate the affected catalogs after adding, renaming or retitling pages.
When a change affects the RDF export, refresh the generated graph documentation
and semantic review before running the docs gate. Follow the
[[docs/development/README#Documentation maintenance|maintenance instructions]]
and the [[docs/okn/README|OKN export guide]].

The replay check uses a capture bound to the N-Quads and join query
hashes. If either changes, prepare a new capture and matching explanation,
preserve the earlier evidence, and update the workflow to replay the new capture.
See the [[docs/okn/evidence/README|capture guide]] and
[[docs/okn/join-support|join explanation guide]].

Keep commits focused and include the observed validation results in the pull
request description.
