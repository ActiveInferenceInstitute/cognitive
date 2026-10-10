---
title: Template Guide
type: guide
status: stable
created: 2024-02-12
updated: 2026-10-10
tags:
  - templates
  - documentation
  - standards
semantic_relations:
  - type: implements
    links:
      - "[[docs/repo_docs/documentation_standards]]"
  - type: relates
    links:
      - "[[docs/templates/template_index|Template index]]"
      - "[[docs/templates/ai_concept_template|AI concept template]]"
      - "[[docs/templates/guide_template|Guide template]]"
---

# Template Guide

How to use the templates in `docs/templates/`. The full list, with the purpose
of each template, is in [[docs/templates/template_index|the template index]].

## Two kinds of template

### Field templates

Eleven `<type>_template.md` files use `{{field}}` entries in their frontmatter and
body: `action`, `agent`, `analysis`, `belief`, `cognitive_architecture`,
`environment`, `experiment`, `goal`, `memory_system`, `observation` and
`reasoning_system`. `cognitive-create-node` renders them with Jinja2 and writes
the result to `knowledge_base/<type>s/<name>.md` (`agents`, `beliefs`, `goals`,
`actions`, `observations` and `environments` are the pluralised folders the tool
knows; other types get an `s` appended):

```bash
cognitive-create-node --type belief --name my_belief --var belief_type=prior
```

Options: `--type` (required), `--name` (required), `--config` (default
`config.yaml`, whose `paths.templates` and `paths.knowledge_base` choose the
folders) and `--var key=value` (repeatable). The tool always supplies
`node_id`, `node_name`, `date` (an ISO timestamp), `type`, `<type>_id` and
`<type>_name`. Any other template field that is not passed with `--var` renders
as empty text, so fill those in by hand afterwards. The tool refuses to
overwrite an existing node.

### Copy-and-edit templates

`ai_concept_template`, `guide_template`, `learning_path_template`,
`linking_template`, `cognitive_concept`, `implementation_example`,
`package_component` and `research_document` are pages that show a structure for
a human to copy. They use `[Bracketed text]`, `{{field}}` or example names for
the parts to replace; rendering them with `cognitive-create-node` only
reproduces the template, so copy the file instead.

## Frontmatter

Every page needs valid YAML frontmatter. The `semantic_relations` links must be
quoted strings:

```yaml
semantic_relations:
  - type: implements
    links:
      - "[[docs/repo_docs/documentation_standards]]"
```

Unquoted `[[...]]` in YAML parses as a nested list, not a link.
`code/scripts/normalize_frontmatter.py --check` lists drifted files and
`--write` fixes them.

## Checking a page

- `cognitive-validate-docs` runs the documentation gate: frontmatter must parse
  and use quoted relation links, Python fences must compile, the forbidden-term
  policy in `docs/policy/documentation_terms.yaml` must pass, and links are
  checked.
- `cognitive-verify-links` checks Obsidian wiki links. By default an
  unresolved extensionless target counts as a concept link;
  `--strict-wiki-links` requires every link to resolve to a file.

Template files legitimately contain example link targets such as
`[[concept1]]`; replace them when you copy the template.

## Code in templates

Code templates are not provided. Runnable examples live in
`docs/examples/README.md`, and the package API is listed in
`docs/api/api_reference.md`.

## Related documentation

- [[docs/templates/ai_concept_template|AI concept template]]
- [[docs/templates/guide_template|Guide template]]
- [[docs/repo_docs/documentation_standards|Documentation standards]]
