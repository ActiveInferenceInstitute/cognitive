---
title: Implementation guides index
type: index
status: stable
created: 2024-02-07
updated: 2026-10-10
tags:
  - implementation
  - guides
  - index
semantic_relations:
  - type: organizes
    links:
      - "[[implementation_guides]]"
      - "[[implementation_patterns]]"
      - "[[docs/api/api_documentation_index]]"
---

# Implementation guides index

| Need | Page |
|---|---|
| Overview of the guides | [[implementation_guides]] |
| Patterns the package follows | [[implementation_patterns]] |
| Build a discrete agent | [[docs/guides/api_implementation]] |
| Choose a runtime for a new agent | [[docs/guides/agent_development]] |
| Active inference agent theory | [[knowledge_base/cognitive/active_inference_agent]] |
| POMDP structure | [[knowledge_base/mathematics/pomdp_structure]] |
| Numerical stability | [[knowledge_base/mathematics/numerical_stability]] |
| File and folder conventions | [[docs/repo_docs/ai_file_organization]], [[docs/repo_docs/ai_folder_structure]], [[docs/repo_docs/naming_conventions]] |
| Documentation style | [[docs/repo_docs/ai_documentation_style]] |
| Validation | [[docs/repo_docs/ai_validation_framework]], [[docs/repo_docs/linking_validation]] |
| Unit testing | [[docs/repo_docs/unit_testing]] |
| Git workflow | [[docs/tools/git_workflow]] |
| Swarm and ant colony material | [[docs/guides/ant_colony_guide]] (documentation only; no ant colony code exists) |
| RxInfer (Julia, external) | [[docs/implementation/rxinfer/README]] |

The package has no integration with PyTorch, TensorFlow, or JAX, and no GPU or
distributed execution path. The runtime dependencies are listed in
`pyproject.toml`.
