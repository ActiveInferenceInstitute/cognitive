---
title: "Emergence From Inference Hierarchies"
type: concept
status: active
created: 2025-01-01
updated: 2025-02-06
tags:
  - free_energy_principle
  - emergence
  - multi_scale
  - downward_causation
  - complexity
  - hierarchical_inference
semantic_relations:
  - type: foundation
    links:
      - [[knowledge_base/free_energy_principle/mathematics/markov_blankets|Markov Blankets]]
      - [[knowledge_base/free_energy_principle/mathematics/advanced_formulations|Advanced Formulations]]
  - type: relates
    links:
      - [[self_organization|Self-Organization]]
      - [[complex_adaptation|Complex Adaptation]]
      - [[knowledge_base/free_energy_principle/cognitive/consciousness|Consciousness]]
      - [[knowledge_base/free_energy_principle/biology/evolution|Evolution]]
---

# Emergence From Inference Hierarchies

## Overview

Emergence -- the appearance of novel properties at higher levels of organization that are not present at lower levels -- is one of the most debated concepts in philosophy of science. The FEP provides a formal framework for understanding emergence through the lens of hierarchical inference and nested Markov blankets.

## Types of Emergence

### Weak Emergence

**Weak emergence** (Bedau, 1997): A property is weakly emergent if it can in principle be derived from lower-level properties but is unexpected or computationally irreducible:

```
Emergent property = F(microscopic properties)
where F is computable but not predictable without full simulation
```

Example: Temperature is weakly emergent from molecular kinetics -- it can be computed but is not a property of any individual molecule.

### Strong Emergence

**Strong emergence**: A property is strongly emergent if it cannot even in principle be derived from lower-level properties -- there is genuine **downward causation** from the higher level to the lower level:

```
Microscopic dynamics at time t+1 depend on macroscopic properties at time t
that are not reducible to microscopic properties at time t
```

Whether strong emergence exists is controversial. The FEP provides tools to analyze this question formally.

### FEP Resolution

The FEP suggests a middle ground through **Markov blanket emergence**:

```
Level n+1 properties emerge when:
1. Level n components form a Markov blanket (collective particular)
2. The collective has its own NESS density
3. The collective minimizes its own free energy
4. This free energy is NOT simply the sum of component free energies
```

The emergent properties are the beliefs, preferences, and dynamics of the collective particular -- properties that have no meaning at the component level but are well-defined at the collective level.

## Nested Markov Blankets and Multi-Scale Organization

### The Nesting Structure

```
Level 0: Atoms       (individual particulars with atomic blankets)
Level 1: Molecules   (collections of atoms with molecular blankets)
Level 2: Organelles  (collections of molecules with organelle blankets)
Level 3: Cells       (collections of organelles with cellular blankets)
Level 4: Tissues     (collections of cells with tissue blankets)
Level 5: Organs      (collections of tissues with organ blankets)
Level 6: Organisms   (collections of organs with organism blankets)
Level 7: Groups      (collections of organisms with group blankets)
Level 8: Societies   (collections of groups with societal blankets)
```

At each level, a new Markov blanket forms from the collective dynamics of the level below. This blanket defines a new "agent" with its own:
- Internal states (collective internal properties)
- Sensory states (collective sensitivity to environment)
- Active states (collective action on environment)
- Free energy (collective divergence from expected states)

### Emergence Criteria

A level n+1 property is **genuinely emergent** (in the FEP sense) if:

```
F_{n+1} != sum_i F_{n,i} + coupling_corrections
```

That is, the collective free energy includes terms that cannot be decomposed into individual free energies plus pairwise coupling. These irreducible terms represent truly emergent properties.

Information-theoretically, this corresponds to **synergy** -- information that is present in the collective but not in any subset of components.

## Downward Causation

### The Problem

Downward causation -- higher-level properties causally influencing lower-level dynamics -- is philosophically problematic because it seems to violate causal closure of physics.

### The FEP Solution

The FEP provides a natural account of downward causation through **empirical priors**:

```
Higher level provides empirical priors for lower level:
p(s_low | s_high) -- higher-level states constrain lower-level inference
```

This is not "spooky" downward causation -- it is the statistical influence of slowly varying macroscopic variables on faster microscopic dynamics. The mechanism is:

1. Microscopic dynamics create macroscopic patterns (upward causation / emergence)
2. Macroscopic patterns constrain microscopic dynamics through boundary conditions (downward causation / top-down prediction)
3. Both are aspects of coupled free energy minimization at different scales

### Example: Neural Downward Causation

```
Level 2 (PFC): Goal representation "reach for cup"
  |-- Provides empirical prior for level 1
Level 1 (Motor cortex): Specific muscle activation pattern
  |-- Provides empirical prior for level 0
Level 0 (Motor neurons): Individual neuron firing
```

The goal (level 2) causally constrains the motor pattern (level 1), which constrains individual neurons (level 0). But this is implemented through ordinary neural signaling -- no mysterious downward force, just hierarchical prediction.

## Emergence of Novel Information

### Information Measures Across Scales

The information content at each scale can be decomposed:

```
I_total = I_individual + I_pairwise + I_triplet + ... + I_synergistic
```

Where:
- `I_individual` = sum of individual components' information
- `I_pairwise` = additional information in pairs
- `I_synergistic` = information only present in the whole system

Emergence corresponds to significant synergistic information -- information that is genuinely new at the higher scale.

### Partial Information Decomposition

The **partial information decomposition** (PID) framework provides tools to quantify emergence:

```
I(X_1, X_2; Y) = Redundancy + Unique_1 + Unique_2 + Synergy
```

Synergy is the genuinely emergent information: it is not available from any individual source but only from the combination.

### FEP and Emergence

Under the FEP, emergence occurs when the collective generative model captures regularities that no individual component's model captures:

```
p_collective(o | s) != product of p_i(o_i | s_i)
```

The collective model has structure -- dependencies, higher-order interactions, contextual effects -- that individual models lack. This structure IS the emergent property.

## The Renormalization Group and Multi-Scale Free Energy

### Scale Transformation

The renormalization group (RG) provides the mathematical framework for relating descriptions at different scales:

```
RG transformation: phi: Level n -> Level n+1
Coarse-graining: Average over microscopic degrees of freedom
Block transformation: Group microscopic variables into macroscopic blocks
```

### Free Energy Under Renormalization

The free energy at each scale is related by:

```
F_{n+1} = R_phi(F_n)  -- the RG transformation of free energy
```

**Fixed points** of the RG flow correspond to scale-invariant descriptions -- systems that look the same at all scales. These are the **universality classes** of statistical physics.

### FEP Universality

The FEP suggests a form of universality in biological systems: despite wildly different microscopic implementations (neurons vs. immune cells vs. gene networks), the macroscopic computational architecture (hierarchical predictive coding, precision-weighted prediction errors, free energy minimization) is universal.

This is because the FEP follows from general principles (ergodicity, Markov blankets, steady state) that do not depend on microscopic details -- just as thermodynamics follows from statistical mechanics regardless of molecular specifics.

## Key References

1. Bedau, M. A. (1997). Weak emergence. *Philosophical Perspectives*, 11, 375-399.
2. Friston, K. (2019). A free energy principle for a particular physics. *arXiv preprint* arXiv:1906.10184.
3. Rosas, F. E., Mediano, P. A. M., Jensen, H. J., Seth, A. K., Barrett, A. B., Carhart-Harris, R. L., & Bor, D. (2020). Reconciling emergences: An information-theoretic approach to identify causal emergence in multivariate data. *PLoS Computational Biology*, 16(12), e1008289.
4. Hoel, E. P. (2017). When the map is better than the territory. *Entropy*, 19(5), 188.
5. Ramstead, M. J. D., Badcock, P. B., & Friston, K. J. (2018). Answering Schrodinger's question. *Physics of Life Reviews*, 24, 1-16.
