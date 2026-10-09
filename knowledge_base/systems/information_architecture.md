---
title: Information Architecture
type: knowledge_base
status: stable
created: 2026-02-07
updated: 2026-10-09
tags:
  - systems
  - active_inference
semantic_relations:
  - type: relates_to
    links:
      - "[[knowledge_base/systems/knowledge_graphs]]"
      - "[[knowledge_base/cognitive/predictive_processing]]"
      - "[[knowledge_base/linking_standards]]"
      - "[[knowledge_base/cognitive/knowledge_organization]]"
---

# Information Architecture

## Overview

Information architecture (IA) is the practice of organising, labelling and structuring information so people can find and understand it. Its classic components, in the formulation of Rosenfeld and Morville, are organisation schemes, labelling, navigation and search. It is an applied discipline with its own methods, and its link to Active Inference is an analogy with some concrete correspondences, not a derivation.

## Main Methods

**Classification.** Hierarchical taxonomies give each item one place; faceted classification, associated with Ranganathan, describes each item along several independent dimensions so it can be reached in more than one way. Hierarchies are easy to browse but force a single parent, while facets handle multi-membership at the cost of a more complex interface.

**Controlled vocabularies and ontologies.** Fixed terms and defined relations reduce ambiguity and let tools reason over the structure. Knowledge graphs generalise this to typed links between entities ([[knowledge_base/systems/knowledge_graphs]]).

**User-centred evaluation.** Card sorting reveals how users group items, and tree testing measures whether they can locate items in a proposed hierarchy. Search logs reveal vocabulary mismatches.

**Navigation and findability.** Labels, breadcrumbs, cross-links and search all serve to let a visitor predict where an item lives.

## Connection to Active Inference

Three correspondences are worth stating, each with a limit.

- *Hierarchy and generative models.* A taxonomy is a nested model of categories, and a user's expectation of where an item sits is a prior over that structure, much as higher levels of a hierarchical model constrain lower ones ([[knowledge_base/cognitive/hierarchical_models]], [[knowledge_base/cognitive/predictive_processing]]). Where the architecture matches the user's prior, navigation is low in prediction error; where it clashes, users get lost. This is a useful lens, not a quantitative model.
- *Foraging.* Information foraging theory (Pirolli and Card 1999) treats browsing as following "information scent" toward patches. Choosing the next link by expected information gain is formally close to epistemic foraging ([[knowledge_base/cognitive/epistemic_foraging]]), though foraging theory predates and is independent of the free-energy principle.
- *Chunking.* Grouping items into a small number of labelled chunks respects working-memory limits and compresses the description of the whole.

## In This Repository

The knowledge base is itself an information architecture. Every folder carries a `README.md` for navigation and an `AGENTS.md` for technical documentation; pages declare typed relations in YAML frontmatter and link to each other with Obsidian-style `[[wiki links]]`; conventions are set out in [[knowledge_base/linking_standards]]. Relevant pages: [[knowledge_base/cognitive/knowledge_organization]], [[knowledge_base/cognitive/machine_readability]], [[knowledge_base/cognitive/plain_text_benefits]] and the ontologies under `knowledge_base/ontology`. Two scripts check the structure (`cognitive-verify-links` for link integrity and `cognitive-validate-docs` for documentation rules), and `NetworkVisualizer` in `code/tools/src/utils/visualization` renders the link graph ([[knowledge_base/cognitive/graph_visualization]]).

## References

- Rosenfeld, L., Morville, P. *Information Architecture for the World Wide Web*. O'Reilly Media.
- Pirolli, P., Card, S. (1999). Information foraging. *Psychological Review*.

## Related Resources

- [[knowledge_base/systems/knowledge_graphs]]
- [[knowledge_base/cognitive/predictive_processing]]
