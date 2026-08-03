---
title: Templates index
type: documentation
status: stable
semantic_relations:
  - type: organizes
    links:
      - '[[AGENTS|Templates instructions]]'
      - '[[template_index|Template index]]'
      - '[[template_guide|Template guide]]'
      - '[[documentation_templates|Documentation templates]]'
      - '[[documentation_templates_index|Documentation templates index]]'
---

# Templates index

This directory provides authoring templates for knowledge-base and
documentation pages.

## Guides

- `template_guide.md` — how to use the templates.
- `template_index.md` — index of all templates.
- `documentation_templates.md` and `documentation_templates_index.md` —
  documentation-specific templates.

## Template files

- Concepts: `ai_concept_template.md`, `cognitive_concept.md`,
  `belief_template.md`, `observation_template.md`, `goal_template.md`,
  `memory_system_template.md`, `reasoning_system_template.md`.
- Agents and environments: `agent_template.md`,
  `cognitive_architecture_template.md`, `environment_template.md`.
- Research: `research_document.md`, `analysis_template.md`,
  `experiment_template.md`.
- Process: `action_template.md`, `guide_template.md`,
  `learning_path_template.md`, `linking_template.md`,
  `implementation_example.md`, `package_component.md`.

## Rules

Templates are starting points: keep frontmatter valid, replace
`semantic_relations` examples with resolving links, and use only real
package symbols in code examples. See `AGENTS.md`.
