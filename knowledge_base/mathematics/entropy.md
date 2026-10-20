---
definition: >-
  Shannon information entropy, the expected negative logarithm of the probability of
  a discrete outcome.
scope_note: >-
  This page treats Shannon entropy and its conditional and joint forms for discrete
  random variables. The correspondences do not identify it with thermodynamic
  entropy, differential entropy or generic disorder.
external_mappings:
  - target: http://www.wikidata.org/entity/Q204570
    relation: exactMatch
    evidence: https://www.wikidata.org/wiki/Q204570
    rationale: >-
      The source gives discrete Shannon entropy as the expected negative log
      probability, with conditional and joint forms. The target defines information
      entropy as expected information delivered by a message. Both denote Shannon
      information entropy, not thermodynamic entropy or a general notion of
      disorder.
    reviewed_on: "2026-10-20"
title: Entropy
description: "Short reference on Shannon, conditional and joint entropy, and its role as the uncertainty measure behind exploration and the ambiguity term of expected free energy."
type: mathematical_concept
status: stable
created: 2025-08-08
tags:
- information-theory
- uncertainty
semantic_relations:
  - type: relates
    links:
      - "[[information_theory]]"
      - "[[mutual_information]]"
      - "[[kl_divergence]]"
  - type: used_by
    links:
      - "[[expected_free_energy]]"
      - "[[action_distribution]]"
      - "[[exploration_exploitation]]"
aliases:
  - shannon_entropy
---


## Entropy

### Shannon entropy

```math

H[X] = - \sum_x p(x)\,\log p(x) \quad (\text{discrete})

```

### Conditional and joint

```math

H[X\mid Y] = H[X,Y] - H[Y]\,,\quad H[X,Y] = -\sum_{x,y} p(x,y)\log p(x,y)

```

### Roles

- Uncertainty measure in exploration: higher H encourages information-seeking

- Appears in EFE via ambiguity terms and expected entropy reductions

### See also

- [[information_theory]] · [[mutual_information]] · [[kl_divergence]]

