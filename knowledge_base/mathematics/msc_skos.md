---

title: Mathematics Subject Classification (MSC) as SKOS

type: concept

status: stable

created: 2025-08-08

tags: [mathematics, ontology, skos, standards, metadata]

semantic_relations:

  - type: relates

    links: [information_theory, graph_theory, optimization_theory]

  - type: integrates

    links: [mathematics_resources, cross_reference_map]

---

## MSC (SKOS)

### Overview

The Mathematics Subject Classification (MSC) represented in SKOS enables standardized tagging, hierarchical browsing, and interoperability across math documents and tools.

### Usage in This Repository

- Tag files with MSC concepts to improve search and discovery

- Align learning paths and indexes with MSC hierarchies

- Map internal topics to external MSC URIs for interoperability

### Tagging example

```yaml

msc:

  primary: 62J12   # Generalized linear models

  secondary: [62F15, 62C10]  # Bayesian inference; decision theory

```

Add to page frontmatter to standardize metadata and enable filters.

### Mapping guidance

- Prefer most specific class that fits the page’s center of gravity

- Use multiple secondary codes to reflect interdisciplinary scope

### Integration

- Links: [[mathematics_resources]] · [[cross_reference_map]]

- Related: [[ontomathpro_ontology]] · [[mathgloss_knowledge_graph]]

### External

- MSC LOD (SKOS): `https://arxiv.org/abs/1204.5086`

