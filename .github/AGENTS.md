---
title: GitHub configuration instructions
type: agents
status: stable
updated: 2026-11-03
---

# AGENTS.md — `.github/`

- Keep Dependabot groups; the two ecosystems update on one weekly pull request
  each.
- Workflow actions are pinned to full commit SHAs with the version in a comment.
  A version bump replaces both.
- Do not add a workflow step that tolerates a missing tool by passing. The
  manuscript and GNN jobs fail when their toolchain cannot be provisioned.
- Parent rules: [../AGENTS.md](../AGENTS.md).
