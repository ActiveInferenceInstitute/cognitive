---
title: Guides index
type: documentation
status: stable
semantic_relations:
  - type: organizes
    links:
      - '[[application/README|Application guides]]'
      - '[[learning_paths/README|Learning paths]]'
      - '[[../examples/README|Examples]]'
      - '[[../implementation/README|Implementation]]'
---

# Guides index

The guides collection combines practical repository guidance with conceptual and
domain-specific material. Working code should begin with the root README and the
executable manuscript; the pages below provide context and navigation.

## Practical entry points

- [`docs/repo_docs/getting_started.md`](../repo_docs/getting_started.md): install,
  verify, and build the repository.
- [`docs/implementation/README.md`](../implementation/README.md): implementation
  documentation and the RxInfer integration subtree.
- [`docs/examples/README.md`](../examples/README.md): executable example commands.
- [`docs/development/README.md`](../development/README.md): development and CI
  gates.

## Application guides

- [`application/README.md`](application/README.md): domain-oriented application
  notes.
- [`application/guide_for_cognitive_modeling.md`](application/guide_for_cognitive_modeling.md)
  and [`application/active_inference_spatial_applications.md`](application/active_inference_spatial_applications.md):
  conceptual application material.

Application guides may be theoretical or domain-specific. They must not imply
that illustrative classes or external environments are shipped by this package.

## Learning paths

- [`learning_paths/README.md`](learning_paths/README.md): learning-path
  navigation.
- [`learning_paths/catalog_of_learning_paths.md`](learning_paths/catalog_of_learning_paths.md):
  catalog of available topical paths.
- [`learning_paths/active_inference_learning_path.md`](learning_paths/active_inference_learning_path.md):
  the general Active Inference path.

## Accuracy boundary

The guide tree includes research and educational notes that are broader than the
implemented Python API. Treat a page as executable documentation only when its
commands and imports are validated by the documentation gate. See
[`AGENTS.md`](AGENTS.md) for the authoring and validation rules.
