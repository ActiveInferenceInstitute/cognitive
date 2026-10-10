---
title: Templates documentation instructions
type: agents
status: stable
created: 2024-01-01
updated: 2026-10-10
tags:
  - templates
  - agents
semantic_relations:
  - type: organizes
    links:
      - "[[docs/templates/README|Templates index]]"
      - "[[docs/templates/template_index|Template index]]"
      - "[[docs/templates/template_guide|Template guide]]"
      - "[[docs/templates/documentation_templates_index|Documentation templates index]]"
---

# Templates documentation instructions

The templates folder provides authoring templates for knowledge-base and
documentation pages. It is also the template directory that
`cognitive-create-node` reads (`paths.templates` in `config.yaml`).

## Usage

- Copy the file and fill in the sections, or render a field template with
  `cognitive-create-node --type <type> --name <name> [--var key=value ...]`.
- Keep the YAML frontmatter valid.
- Replace every `semantic_relations` example with quoted links that resolve to
  real files, for example `- "[[docs/api/api_reference|API reference]]"`.
- `template_guide.md` explains when to use each kind of template.

## Rules

- A file named `<type>_template.md` is a `cognitive-create-node` template: it
  is rendered with Jinja2, so `{{field}}` entries are substituted and any
  unset field renders as empty text. Do not add `{%` or `{#` sequences to such
  a file.
- Templates must not contain fictional fields or APIs. Where a template shows
  a code example, it must use symbols from the installed package.
- New templates must be added to `template_index.md` and `README.md`.
- Do not use the templates to define classes that do not exist in the
  repository.
