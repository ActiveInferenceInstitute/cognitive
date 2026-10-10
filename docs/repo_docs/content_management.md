---
title: Content management
type: guide
status: stable
created: 2024-02-06
updated: 2026-10-10
tags:
  - content
  - maintenance
  - organization
semantic_relations:
  - type: implements
    links:
      - "[[docs/repo_docs/documentation_standards|Documentation standards]]"
  - type: relates
    links:
      - "[[docs/repo_docs/validation|Validation]]"
      - "[[docs/repo_docs/folder_structure|Folder structure]]"
---

# Content management

How documentation is created, changed, reviewed, and retired. Git is the only
version store; there is no separate archive, review-assignment system, or
scheduled content audit.

## Create

1. Pick the folder from [[docs/repo_docs/folder_structure|Folder structure]].
2. Start from a template when one fits: `cognitive-create-node --type <type>
   --name <name>` renders `docs/templates/<type>_template.md` into
   a `knowledge_base/` subfolder named for the type (see
   [[docs/templates/template_guide|Template guide]]). Otherwise copy the
   frontmatter pattern from a neighbouring page.
3. Fill every field; the validator rejects frontmatter left at a template value
   (`unique_identifier`, `timestamp`, `lorem_ipsum`).
4. Add `semantic_relations` and a closing "Related" section
   ([[docs/repo_docs/linking_completeness|Linking completeness]]).

## Change

- Edit the page and set `updated:` to the date of the change.
- Keep one authoritative page per fact and link to it. When code changes, the
  page that describes it changes in the same commit, and the snippets in the API
  reference are re-run against the tree.
- Rewrite stale text rather than annotating it. A page should read as a
  present-state description, not as a log of how it got there.
- Record notable behaviour in `CHANGELOG.md` under `[Unreleased]`; a change to a
  computed result is marked **behaviour change**. Outstanding work goes in the
  tracked backlog file at the repository root, not in page text (the forbidden
  term policy rejects work-in-progress markers).

## Verify

```bash
python code/scripts/kb_index.py --write .          # refresh generated catalogs
python code/scripts/normalize_frontmatter.py --write .
python code/scripts/validate_docs.py --json
```

`validate_docs.py` covers frontmatter, link resolution, `python` fence syntax,
forbidden terms, and catalog freshness. See
[[docs/repo_docs/ai_validation_framework|Validation framework]].

## Review

Changes are reviewed as ordinary pull requests, with the `quality` workflow
(`.github/workflows/quality.yml`) as the automated reviewer. Reviewers check
what the scripts cannot: whether a claim is true and whether a link points at the
right page. Periodic audits are written up as `REVIEW_LOG_YYYY-MM-DD.md` files
at the repository root.

## Rename, move, and retire

- Rename or move a page with `git mv`, then fix inbound links: run
  `python code/scripts/verify_links.py .` and correct each `source` it lists.
  Obsidian rewrites links on rename only when the rename happens inside
  Obsidian.
- To retire a page, delete it in a commit that also removes or redirects every
  inbound link. Git history is the archive; do not leave dead copies in the
  tree or add `archive/` folders for pages.
- A rename that changes a heading breaks `[[page#heading]]` anchors; the link
  checkers report these as warnings.

## Related documentation

- [[docs/repo_docs/documentation_standards|Documentation standards]]
- [[docs/repo_docs/validation|Validation]]
- [[docs/repo_docs/folder_structure|Folder structure]]
