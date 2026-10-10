---
title: Git workflow
type: tool
status: stable
created: 2024-07-25
updated: 2026-10-10
tags:
  - git
  - workflow
  - tools
semantic_relations:
  - type: relates
    links:
      - "[[docs/tools/git_tools|Git tooling]]"
      - "[[docs/development/contribution_guide|Contribution guide]]"
      - "[[docs/tools/ci_tools|CI tools]]"
---

# Git workflow

## Branches

Development happens on `main`, which tracks `origin/main` on GitHub
(`ActiveInferenceInstitute/cognitive`). The repository does not use
`develop`, `release/*`, or `hotfix/*` branches. The contribution guide asks
for a pull request description with the observed validation results.

## Commit messages

Recent history uses a `type(scope): summary` subject, for example:

```text
ci: install pandoc and pandoc-crossref from pinned release binaries
feat(models): policy_aggregation and pinned dispatcher semantics
feat(tests): mutation gate over the corrected mathematics
docs: quote semantic_relations links and restore fenced link examples
test: cover Simple_POMDP edge paths and verify_links CLI
```

Older commits use free-form subjects. No tool enforces the format; there is
no commit-message hook.

## Before opening a pull request

Run the checks listed in
[[docs/development/contribution_guide|the contribution guide]] and paste the
observed results into the pull request description. Keep commits focused: a
change to runtime behavior carries its regression test, and a change to the
mathematical contract updates the executable manuscript.

## What gates a push

The `quality` workflow (`.github/workflows/quality.yml`) runs on every push and
pull request. See [[docs/tools/ci_tools|CI tools]] for its jobs. Because
`validate_docs.py` reads `git ls-files`, stage new documentation before running
it locally.
