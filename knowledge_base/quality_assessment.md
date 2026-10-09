---
title: Knowledge Base Quality Assessment
type: assessment
status: active
created: 2024-02-07
updated: 2026-10-09
tags:
  - quality
  - assessment
  - maintenance
semantic_relations:
  - type: relates
    links:
      - "[[knowledge_base/catalog]]"
      - "[[knowledge_base/linking_standards]]"
      - "[[knowledge_base/AGENTS]]"
---

# Knowledge Base Quality Assessment

A measured account of the vault's state and its known gaps. The figures come
from the repository's own gates and can be reproduced; they are not estimates.
Nothing here is a target date or a rating.

## Size

About 800 pages across eleven domains. Exact per-domain counts are in each
domain's generated `catalog.md`, which lists every page with its title and
opening line; [[knowledge_base/catalog]] links them all. The largest domains
are cognitive science (about 290 pages) and mathematics (about 200).

## What the gates enforce

| Check | Command | Guarantees |
|-------|---------|------------|
| Explicit wiki links | `python code/scripts/verify_links.py .` | Every path-style `[[link]]` resolves |
| Standard Markdown links | `python code/scripts/check_markdown_links.py .` | Every `[text](path)` link and image resolves |
| Frontmatter | `python code/scripts/validate_docs.py` | Frontmatter parses, no template values, `semantic_relations` links are quoted wiki links |
| Catalogs | `python code/scripts/kb_index.py --check` | Every page appears in its domain catalog |
| Term policy | `docs/policy/documentation_terms.yaml` | No page advertises content that does not exist |
| Python examples | `validate_docs` | Fenced `python` blocks compile |

## Known gaps

**Unresolved concept links.** About 9,800 `[[concept]]` links, to about 7,300
distinct targets, have no page behind them. The link checker treats extensionless
names that do not resolve as concept links and skips them, so they do not fail
the gate. Most are single-use names produced when a page was written (for
example a "Related" list naming topics the author never wrote up). The most
referenced ones now have pages; a few hundred are still linked three or more
times. Some are not concepts at all and should be unlinked rather than written:
`new_methods`, `validation_framework`, `version_control`.

**Descriptions.** About fifty pages open with a formula or a list and therefore
appear in their catalog by title only. Giving them a one-sentence `description:`
in frontmatter fixes the catalog entry and any search snippet.

**Cited works.** Reference lists on pages written or expanded in the 2026-10-09
pass were compiled from memory and have not been checked against the sources.
Verify author, year, title, and venue before citing a page in a publication.

**Scholarship review.** Several hundred older cognitive-science pages were
generated in bulk and carry broad claims with few citations. Page-by-page
editorial review is a standing maintenance activity, not a one-off task.

## How to improve a page

1. Replace general statements with the specific mechanism, equation, or result.
2. State how strongly the link to Active Inference is established: a derivation,
   an active research programme, or an analogy.
3. Link only to pages that exist, with `[[stem]]` or `[[knowledge_base/<domain>/<stem>]]`.
4. Keep `semantic_relations` links as quoted strings (`"[[page]]"`).
5. Regenerate the catalogs with `python code/scripts/kb_index.py --write` and
   run `python code/scripts/validate_docs.py`.
