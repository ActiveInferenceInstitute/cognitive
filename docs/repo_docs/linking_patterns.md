---
title: Linking patterns
type: guide
status: stable
created: 2024-02-12
updated: 2026-10-10
tags:
  - obsidian
  - linking
  - organization
semantic_relations:
  - type: implements
    links:
      - "[[docs/repo_docs/documentation_standards|Documentation standards]]"
  - type: relates
    links:
      - "[[docs/repo_docs/obsidian_usage|Obsidian usage]]"
      - "[[docs/repo_docs/folder_structure|Folder structure]]"
---

# Linking patterns

How pages in this vault link to each other, taken from the existing pages. The
syntax and resolution rules are in
[[docs/repo_docs/obsidian_linking|Obsidian linking]]; the checkers are in
[[docs/repo_docs/linking_validation|Link validation]].

## Concept pages in `knowledge_base/`

A concept page declares its neighbours twice: in frontmatter, as typed
relations, and at the end of the body, as a "Related concepts" list with a short
reason for each link. For example, `knowledge_base/cognitive/action_planning.md`
ends with:

```markdown
## Related concepts

- [[planning_as_inference]] — the general formulation
- [[knowledge_base/mathematics/expected_free_energy]] — the score being summed
```

Bare names such as `[[planning_as_inference]]` are the common form for links
inside one domain. Use the full path when the target is in another domain, when
the name could match more than one page, or when the link must be enforced by
the gate.

## Hub pages

Each domain has a `README.md` that introduces it and a generated `catalog.md`
that lists every page. Link to a hub rather than copying its list. The vault's
entry points are `knowledge_base/index.md`, `knowledge_base/README.md`, and
`knowledge_base/catalog.md`.

## Documentation pages

A guide links forward to the reference it depends on and back to the standard it
follows. API pages link to [[docs/api/api_reference|the API reference]]; agent
pages link to [[docs/agents/agent_docs_readme|the agent clearinghouse]]. Do not
describe the same API in two pages; link to the one authoritative page.

## Code references

Documentation names code in code spans (`code/tools/src/models/active_inference/`,
`DiscreteGenerativeModel`) rather than wiki links, because the link checker
indexes Markdown and a code path is not a page. Link to a folder's `README.md`
or `AGENTS.md` when you mean the folder's documentation, for example
`[[code/Things/Simple_POMDP/AGENTS]]`.

## Frontmatter relations

```yaml
semantic_relations:
  - type: relates
    links:
      - "[[knowledge_base/cognitive/action_planning|Action planning]]"
```

The links must be quoted. Common `type` values are `relates`, `implements`,
`extends`, `organizes`, and `documents`.

## Reciprocity

Links are not required to be reciprocal, and no tool checks for it. Obsidian
shows backlinks for every inbound link automatically, so add a reverse link only
when a reader of the target page would want to be sent back.

## Related documentation

- [[docs/repo_docs/obsidian_linking|Obsidian linking]]
- [[docs/repo_docs/linking_completeness|Linking completeness]]
- [[docs/repo_docs/linking_validation|Link validation]]
