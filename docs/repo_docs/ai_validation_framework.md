---
title: Validation framework
type: guide
status: stable
created: 2024-02-06
updated: 2026-10-10
tags:
  - validation
  - quality
  - documentation
semantic_relations:
  - type: implements
    links:
      - "[[docs/repo_docs/documentation_standards|Documentation standards]]"
  - type: relates
    links:
      - "[[docs/repo_docs/validation|Validation]]"
      - "[[docs/repo_docs/linking_validation|Link validation]]"
---

# Validation framework

Documentation quality is enforced by five scripts in `code/scripts/`, each also
installed as a console command. There are no quality scores, embedding checks,
or threshold files; every check is pass or fail.

| Script | Console command | Checks |
|---|---|---|
| `validate_docs.py [root] [--json]` | `cognitive-validate-docs` | The aggregate gate (below). |
| `verify_links.py [root] [--strict-wiki-links] [--json]` | `cognitive-verify-links` | Obsidian `[[wiki links]]`. |
| `check_markdown_links.py [root] [--json]` | none | Standard `[text](target)` links and anchors. |
| `normalize_frontmatter.py [root] (--check or --write)` | `cognitive-normalize-frontmatter` | `semantic_relations` links are quoted strings. |
| `kb_index.py [root] (--check or --write)` | `cognitive-kb-index` | Generated `catalog.md` files are current. |

Each exits non-zero when it finds a problem (`--check` for the last two).

## What `validate_docs.py` checks

It reads the files returned by `git ls-files`, so untracked files are not
scanned, and reports `forbidden_terms`, `documentation_errors`, `links`,
`markdown_links`, and `manuscript` sections in its JSON.

1. **Forbidden terms.** Every `.md`, `.py`, `.yaml`, `.yml`, `.toml`, `.txt`,
   `.json`, and `.sh` file is scanned for the whole-word, case-insensitive terms
   in `docs/policy/documentation_terms.yaml`. The files that must spell the terms
   are listed there as exempt.
2. **Frontmatter.** A file that starts with `---` must have a non-empty YAML
   mapping, no template values (`unique_identifier`, `timestamp`,
   `lorem_ipsum`), and quoted `semantic_relations` links. A Markdown file that
   does not start with `---` is not checked for frontmatter.
3. **Python fences.** Fences tagged `python` or `py` must compile. They are not
   executed.
4. **Wiki links.** The `verify_links.py` report in default mode: explicit file
   references that do not resolve are errors.
5. **Markdown links.** The `check_markdown_links.py` report: broken file targets
   are errors.
6. **Catalogs.** Missing or stale knowledge-base catalogs are errors.
7. **Exports.** Every name in `cognitive.__all__` must exist on the package.
8. **Manuscript.** `docs/manuscript/` must contain the eleven numbered sections
   `00` to `09` and `99`, with no duplicate `{#eq:}`, `{#fig:}`, `{#tbl:}`, or
   `{#sec:}` labels, no unresolved `[@fig:...]`-style references, every citation key
   present in `references.bib`, every figure path produced by
   `code/scripts/build_manuscript.py`, and a `config.yaml` with an `experiment`
   mapping.

Anchor mismatches in either link checker are warnings and do not fail the gate.

## Where the gates run

CI (`.github/workflows/quality.yml`, the only workflow) runs
`validate_docs.py --json` (which includes the link checks) on every push and pull
request, in the `checks` job. Locally, run the same commands from
the repository root; the full development loop is in
[`docs/development/README.md`](../development/README.md). The repository has no
pre-commit hook configuration.

## Fixing a failure

- Forbidden term: reword the sentence so it states what exists. Do not swap in
  a synonym for the same claim.
- Unquoted relation link: `python code/scripts/normalize_frontmatter.py --write .`
- Stale catalog: `python code/scripts/kb_index.py --write .`
- Broken link: correct the target, or write a bare concept name if no page is
  meant (see [[docs/repo_docs/linking_validation|Link validation]]).

## Related documentation

- [[docs/repo_docs/validation|Validation]]
- [[docs/repo_docs/linking_validation|Link validation]]
- [[docs/repo_docs/documentation_standards|Documentation standards]]
- [[docs/tools/doc_validator|Documentation validator notes]]
