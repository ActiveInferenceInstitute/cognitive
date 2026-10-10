---
title: Documentation Templates
type: index
status: stable
created: 2024-02-06
updated: 2026-10-10
tags:
  - templates
  - documentation
  - standards
semantic_relations:
  - type: relates
    links:
      - "[[docs/templates/template_index|Template index]]"
      - "[[docs/templates/template_guide|Template guide]]"
      - "[[docs/repo_docs/documentation_standards|Documentation standards]]"
---

# Documentation Templates

This page says which template to start from for each kind of documentation
page. Every template named here exists in `docs/templates/`; the complete list
is in [[docs/templates/template_index|the template index]].

| You are writing | Start from |
|-----------------|------------|
| A concept page | [[docs/templates/ai_concept_template|AI concept template]] or [[docs/templates/cognitive_concept|Cognitive concept template]] |
| An instructional guide | [[docs/templates/guide_template|Guide template]] |
| A learning path | [[docs/templates/learning_path_template|Learning path template]] |
| A research page | [[docs/templates/research_document|Research document template]] |
| A worked implementation | [[docs/templates/implementation_example|Implementation example template]] |
| A package component page | [[docs/templates/package_component|Package component template]] |
| A page that needs a consistent set of links | [[docs/templates/linking_template|Linking template]] |
| An experiment or its analysis | [[docs/templates/experiment_template|Experiment template]], [[docs/templates/analysis_template|Analysis template]] |
| A model component (agent, belief, action, goal, observation, environment, memory, reasoning, architecture) | the matching `<type>_template.md`, rendered with `cognitive-create-node` |

Frontmatter for every page follows
[[docs/repo_docs/documentation_standards|the documentation standards]]:
`title`, `type`, `status`, `created`, `updated`, optional `tags`, and
`semantic_relations` whose links are quoted strings. Instructions for copying
and rendering templates are in [[docs/templates/template_guide|the template guide]].
