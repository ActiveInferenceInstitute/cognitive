---
title: Documentation validator
type: tool
status: stable
updated: 2026-10-10
created: 2026-08-02
tags:
  - validation
  - documentation
  - tools
semantic_relations:
  - type: documents
    links:
      - "[[../../code/scripts/validate_docs.py|Validator source]]"
---

# Documentation validator

`code/scripts/validate_docs.py` is the repository's documentation gate:

```bash
python code/scripts/validate_docs.py --json
```

It checks YAML frontmatter validity, Python fenced blocks (compiled, not
executed), public
package exports, manuscript structure/labels/citations/figures, forbidden
terms, explicit wiki links, and standard Markdown links. Companion scripts:

- `code/scripts/verify_links.py . --json` — explicit wiki-link audit.
- `code/scripts/check_markdown_links.py . --json` — standard Markdown link
  and anchor audit (anchor mismatches are advisory warnings).
- `code/scripts/normalize_frontmatter.py --check .` and
  `code/scripts/kb_index.py --check .` — frontmatter quoting and catalog
  freshness (both are also reported by the validator).

The validator reads `git ls-files`, so only tracked files are scanned. The
forbidden-term list and its exempt files come from
`docs/policy/documentation_terms.yaml`; see
[[docs/policy/README|Repository policy files]].
