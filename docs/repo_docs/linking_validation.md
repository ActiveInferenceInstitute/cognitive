---
title: Link validation
type: guide
status: stable
created: 2024-02-06
updated: 2026-10-10
tags:
  - validation
  - linking
  - documentation
semantic_relations:
  - type: implements
    links:
      - "[[docs/repo_docs/linking_completeness|Linking completeness]]"
      - "[[docs/repo_docs/ai_validation_framework|Validation framework]]"
  - type: extends
    links:
      - "[[docs/repo_docs/obsidian_linking|Obsidian linking]]"
---

# Link validation

Two scripts check links, and `validate_docs.py` runs both.

## Wiki links: `verify_links.py`

```bash
python code/scripts/verify_links.py .                       # human-readable report
python code/scripts/verify_links.py . --json                # machine-readable report
python code/scripts/verify_links.py . --strict-wiki-links   # also fail on unresolved bare names
```

Also installed as `cognitive-verify-links`. The JSON report has the keys
`checked_links`, `resolved_links`, `skipped_concept_links`,
`indexed_markdown_files`, `anchor_warnings`, and `broken_links` (each entry has
`source`, `link`, and `target`). The exit status is 1 when `broken_links` is not
empty. Anchor warnings never change the exit status. Resolution rules are in
[[docs/repo_docs/obsidian_linking|Obsidian linking]].

## Standard Markdown links: `check_markdown_links.py`

```bash
python code/scripts/check_markdown_links.py . --json
```

It checks `[text](target)` and `![alt](target)` links. File targets must exist
relative to the page or the repository root; `#anchor` fragments are matched
against heading slugs and reported as warnings. External URLs are not fetched.
It skips the manuscript's generated `figures/` references. The report has
`checked_links`, `broken_links`, `anchor_warnings`, and `indexed_markdown_files`,
and the exit status is 1 when a link is broken.

## Triage

1. Run `verify_links.py . --json` and group `broken_links` by `source`.
2. If the target page exists under a different path, correct the link. If the
   page was renamed, update every inbound link in the same change.
3. If no page is meant, replace the link with plain text or a code span, or
   (for a concept you intend to write) a bare name without a path or suffix.
4. For a wider audit, run with `--strict-wiki-links`. Expect many results: bare
   concept links are allowed by convention, so strict mode is a worklist, not a
   gate.

A small script can summarize a report; this one groups strict-mode failures by
source file:

```python
from collections import Counter

from scripts.verify_links import verify_link_report

report = verify_link_report(".", strict_wiki_links=True)
print(f"checked={report.checked_links} resolved={report.resolved_links}")
by_source = Counter(item["source"] for item in report.broken_links)
for source, count in by_source.most_common(3):
    print(count, source)
```

## Gates

CI runs `validate_docs.py --json` and, as a separate step,
`verify_links.py . --json`. The tests are `code/tests/test_verify_links.py`,
`test_check_markdown_links.py`, and `test_validate_docs.py`. Nothing validates
links to external websites.

## Related documentation

- [[docs/repo_docs/ai_validation_framework|Validation framework]]
- [[docs/repo_docs/linking_analysis|Linking analysis]]
- [[docs/tools/link_checker|Link checker notes]]
