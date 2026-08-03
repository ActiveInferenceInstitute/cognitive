---
title: Templates documentation instructions
type: agents
status: stable
created: 2024-01-01
updated: 2026-08-02
tags:
  - templates
  - agents
  - reusable
  - patterns
semantic_relations:
  - type: organizes
    links:
      - '[[README|Templates index]]'
      - '[[template_index|Template index]]'
      - '[[template_guide|Template guide]]'
      - '[[documentation_templates_index|Documentation templates]]'
---

# Templates documentation instructions

The templates folder provides authoring templates for knowledge-base and
documentation pages.

## Usage

- Templates are starting points: copy the file, fill in the sections, and
  keep the YAML frontmatter valid.
- Every template ships with `semantic_relations` examples; replace them with
  links that resolve to real files.
- The template guides (`template_guide.md`, `documentation_templates.md`)
  explain when to use each template.

## Rules

- Templates must not contain fictional fields or APIs. Where a template shows
  a code example, it must use symbols from the installed package.
- New templates must be added to `template_index.md` and `README.md`.
- Do not use the templates to define classes that do not exist in the
  repository.
