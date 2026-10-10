---
title: Templates index
type: documentation
status: stable
created: 2024-01-01
updated: 2026-10-10
semantic_relations:
  - type: organizes
    links:
      - "[[docs/templates/AGENTS|Templates instructions]]"
      - "[[docs/templates/template_index|Template index]]"
      - "[[docs/templates/template_guide|Template guide]]"
      - "[[docs/templates/documentation_templates|Documentation templates]]"
      - "[[docs/templates/documentation_templates_index|Documentation templates index]]"
---

# Templates index

This directory holds the authoring templates for knowledge-base and
documentation pages. `config.yaml` points `paths.templates` here, and
`cognitive-create-node` renders the `<type>_template.md` files from this
directory into `knowledge_base/`.

## Guides

- `template_guide.md` explains the two kinds of template and how to use them.
- `template_index.md` lists every template with its purpose.
- `documentation_templates.md` and `documentation_templates_index.md` cover the
  documentation-page templates.

## Template files

- Concepts: `ai_concept_template.md`, `cognitive_concept.md`,
  `belief_template.md`, `observation_template.md`, `goal_template.md`,
  `memory_system_template.md`, `reasoning_system_template.md`.
- Agents and environments: `agent_template.md`,
  `cognitive_architecture_template.md`, `environment_template.md`.
- Research: `research_document.md`, `analysis_template.md`,
  `experiment_template.md`.
- Process and reference: `action_template.md`, `guide_template.md`,
  `learning_path_template.md`, `linking_template.md`,
  `implementation_example.md`, `package_component.md`.

## Rules

Templates are starting points: keep frontmatter valid, replace
`semantic_relations` examples with resolving quoted links, and use only real
package symbols in code examples. See `AGENTS.md`.
