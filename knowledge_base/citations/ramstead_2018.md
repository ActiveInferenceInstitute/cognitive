---
title: "Answering Schrodinger's Question: A Free-Energy Formulation"
authors:
  - "Maxwell J. D. Ramstead"
  - "Paul B. Badcock"
  - "Karl J. Friston"
type: citation
status: verified
created: 2025-01-01
year: 2018
journal: "Physics of Life Reviews"
volume: 24
pages: 1-16
doi: "10.1016/j.plrev.2017.09.001"
tags:
  - free_energy
  - variational_ecology
  - niche_construction
  - self_organization
  - evolution
semantic_relations:
  - type: foundational_for
    links:
      - [[knowledge_base/biology/evolutionary_dynamics]]
      - niche construction
  - type: extends
    links:
      - [[friston_2013]]
      - [[friston_2010]]
  - type: cited_by
    links:
      - [[constant_2018]]
      - [[kirchhoff_2018]]
---

# Answering Schrodinger's Question: A Free-Energy Formulation

## Authors
- **Maxwell J. D. Ramstead** (McGill University / VERSES Research)
- **Paul B. Badcock** (University of Melbourne)
- **Karl J. Friston** (UCL)

## Publication Details
- **Journal**: Physics of Life Reviews
- **Year**: 2018
- **Volume**: 24
- **Pages**: 1-16
- **DOI**: [10.1016/j.plrev.2017.09.001](https://doi.org/10.1016/j.plrev.2017.09.001)

## Abstract
This paper answers Schrodinger's famous question "What is life?" using the free energy principle. The authors develop a "variational ecology" -- a framework for understanding how living systems at all scales maintain their organization by minimizing variational free energy. They extend the FEP to evolutionary and ecological scales, arguing that niche construction, natural selection, and ecological dynamics can all be understood as different timescales of free energy minimization. The paper provides a multi-scale account from cells to cultures.

## Key Contributions

### Variational Ecology
- **Multi-Scale Framework**: Free energy minimization across biological scales
- **Niche Construction**: Organisms shape environments to minimize surprise
- **Cultural Evolution**: Cultural practices as collective free energy minimization
- **Eco-Evo-Devo**: Integrating ecology, evolution, and development

### Answering Schrodinger
- **What is Life?**: Self-organizing systems that minimize free energy
- **Negative Entropy**: Organisms maintain order by bounding surprise
- **Markov Blankets**: Defining boundaries of living systems at every scale
- **Adaptive Fitness**: Fitness as model evidence (negative free energy)

### Multi-Scale Active Inference
- **Phylogenetic**: Natural selection over evolutionary timescales
- **Ontogenetic**: Development and learning over a lifetime
- **Epigenetic**: Gene expression and regulation
- **Behavioral**: Action and perception in real time
- **Cultural**: Social practices and institutions

## Core Concepts

### Variational Niche Construction
Organisms do not merely adapt to environments; they actively construct their niches to minimize free energy:

```
F_total = F_internal + F_external
```

Where:
- `F_internal`: Free energy minimized through internal model updating (perception, learning)
- `F_external`: Free energy minimized through acting on the environment (niche construction)

### Multi-Scale Markov Blankets
Living systems are nested Markov blankets:
- **Cell Level**: Cell membrane separates internal from external
- **Organism Level**: Sensory surfaces and motor effectors
- **Social Level**: Social norms and cultural boundaries
- **Ecosystem Level**: Ecological niches and trophic structures

### Adaptive Fitness as Model Evidence
Natural selection is recast as variational inference:
```
Fitness = ln p(y|m) approx -F
```

Organisms with better generative models (lower free energy) are fitter.

## Mathematical Formalism

### Free Energy Across Scales
At each biological scale, the same principle applies:
```
F_scale = E_q[ln q(mu) - ln p(y, mu|m)]
```

But the variables change:
- **Cellular**: `mu` = metabolic states, `y` = chemical signals
- **Neural**: `mu` = neural activity, `y` = sensory input
- **Behavioral**: `mu` = beliefs about environment, `y` = outcomes
- **Evolutionary**: `mu` = phenotype, `y` = environmental pressures

## Impact and Applications

### Theoretical Biology
- **Unified Biology**: Single principle across all biological scales
- **Evolutionary Theory**: Information-theoretic view of natural selection
- **Ecology**: Ecosystem dynamics as multi-agent inference
- **Developmental Biology**: Morphogenesis through free energy minimization

### Social Science
- **Cultural Evolution**: Social norms as shared generative models
- **Institutional Design**: Organizations as free energy minimizing structures
- **Anthropology**: Cultural practices as niche construction

### Philosophy
- **Philosophy of Biology**: New framework for understanding life
- **Teleology**: Purpose without intention through free energy minimization
- **Emergence**: How complexity arises from simple principles

## Related Work

### Foundational Papers
- [[friston_2013]] - Life as we know it
- [[friston_2010]] - Free energy principle review
- [[friston_2019_particular]] - Particular physics

### Related Extensions
- [[constant_2018]] - Variational niche construction
- [[kirchhoff_2018]] - Markov blankets of life
- [[friston_2015_knowing]] - Pattern regulation

## Citations and Influence
This paper has been widely cited as a key extension of the FEP to biological and ecological scales. It established the concept of "variational ecology" and provided the theoretical foundation for applying the FEP to evolution, niche construction, and cultural dynamics. It bridges the gap between the neuroscience-focused FEP literature and broader biological theory.

## Reading Guide
1. **Introduction**: Schrodinger's question and the FEP
2. **Variational Ecology**: Multi-scale free energy minimization
3. **Niche Construction**: Active shaping of the environment
4. **Evolution**: Natural selection as inference
5. **Culture**: Social and cultural extensions

---

> **Variational Ecology**: Extends the free energy principle to evolutionary and ecological scales, answering "What is life?"

---

> **Multi-Scale**: Demonstrates how the same principle applies from cells to cultures through nested Markov blankets.

---

> **Niche Construction**: Formalizes how organisms actively shape their environments as part of free energy minimization.
