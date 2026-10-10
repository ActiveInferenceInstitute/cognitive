---
title: Git tooling
type: tool
status: stable
created: 2024-07-25
updated: 2026-10-10
tags:
  - git
  - tools
  - version-control
semantic_relations:
  - type: relates
    links:
      - "[[docs/tools/git_workflow|Git workflow]]"
      - "[[docs/tools/ci_tools|CI tools]]"
      - "[[docs/development/contribution_guide|Contribution guide]]"
---

# Git tooling

The repository uses plain Git. It ships no Git extensions, hooks, attributes
files, or synchronization utilities of its own: the checked-in `.git/hooks`
directory holds only Git's sample files, there is no `.gitattributes`, no Git
LFS configuration, and no `.pre-commit-config.yaml`.

## What Git provides to the gates

- `code/scripts/validate_docs.py` scans the files returned by
  `git ls-files -z`, so only tracked files are checked for frontmatter,
  fenced Python, links, and the forbidden-term policy. A new file must be
  `git add`-ed before the gate sees it.
- `.gitignore` excludes Python caches, build trees, virtual environments, and
  generated output. Figures, animations, benchmark JSON, and manuscript
  builds go to temporary or caller-selected directories and are not committed.

## Checks to run before committing

The enforced gates live in `.github/workflows/quality.yml` and are run by
hand with the commands in [[docs/development/README|Development resources]]:

```bash
python -m pytest -q --cov
ruff check .
mypy code/tools/src code/Things code/scripts
python code/scripts/validate_docs.py --json
python code/scripts/verify_links.py . --json
```

There is no commit hook that runs them; the `quality` workflow runs them on
every push and pull request.

## Obsidian and Git

The `.obsidian/` folder holds vault configuration, and its only community
plugin is `3d-graph`. There is no Obsidian Git plugin configured, so commits
are made from the command line or another Git client.

## Related

- [[docs/tools/git_workflow|Git workflow]] for branch and commit conventions.
- [[docs/tools/ci_tools|CI tools]] for the workflow that gates pushes.
