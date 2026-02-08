---
title: "Evolution as Free Energy Minimization Across Generations"
type: concept
status: active
created: 2025-01-01
updated: 2025-02-06
tags:
  - free_energy_principle
  - evolution
  - natural_selection
  - niche_construction
  - model_evidence
  - fitness
semantic_relations:
  - type: foundation
    links:
      - [[knowledge_base/free_energy_principle/mathematics/core_principle|Core Principle]]
      - [[knowledge_base/free_energy_principle/mathematics/markov_blankets|Markov Blankets]]
  - type: relates
    links:
      - [[development|Development]]
      - [[ecology|Ecology]]
      - [[homeostasis|Homeostasis]]
      - [[knowledge_base/free_energy_principle/systems/complex_adaptation|Complex Adaptation]]
---

# Evolution as Free Energy Minimization Across Generations

## Overview

The FEP extends beyond individual organisms to evolution itself. Under this view, natural selection is a process of free energy minimization across generations: organisms whose generative models are well-fitted to their environment (high model evidence) leave more descendants, and the population-level distribution of generative models converges on those that minimize free energy.

This perspective unifies evolutionary biology with Bayesian inference, casting fitness as model evidence and natural selection as Bayesian model selection over generations.

## Fitness as Model Evidence

### The Core Mapping

The central insight is the formal equivalence between biological fitness and Bayesian model evidence:

```
Fitness(organism) ~ p(o | m) = model evidence
                  = integral p(o | s, m) * p(s | m) ds
```

Where:
- `o` = environmental observations (challenges, resources, mates)
- `s` = hidden environmental states
- `m` = the organism's generative model (its phenotype, including physiology and behavior)

An organism with a good generative model:
- Predicts its environment accurately (low surprisal)
- Maintains homeostasis (stays within viable states)
- Reproduces successfully (satisfies prior preferences)
- Has high fitness (high model evidence)

### The Free Energy Bound

Since free energy upper-bounds surprisal:
```
F >= -ln p(o | m)
```

An organism that minimizes free energy also maximizes (bounds) model evidence, which is equivalent to maximizing fitness.

**Over an organism's lifetime**:
```
Lifetime fitness ~ exp(-integral F(t) dt)  -- integral of free energy over lifetime
```

Organisms with chronically low free energy (good predictions, maintained homeostasis, successful behavior) have higher lifetime fitness.

### Population-Level Selection

At the population level, natural selection acts on the distribution of generative models:

```
p(m | generation_n+1) proportional to p(o | m) * p(m | generation_n)
```

This is Bayesian model selection: the posterior distribution over models (next generation) is proportional to the model evidence (fitness) times the prior (current generation). Models with high evidence are selected for; those with low evidence are selected against.

## Niche Construction as Active Inference

### The Extended Phenotype

Niche construction -- organisms modifying their environment -- is the evolutionary analog of active inference:

```
Individual active inference: Change o to match predictions (within a lifetime)
Niche construction: Change the environment to match phenotypic expectations (across generations)
```

Examples:
- **Beaver dams**: Beavers construct an environment that matches their aquatic generative model
- **Termite mounds**: Termites create a microclimate matching their homeostatic preferences
- **Human culture**: Humans construct social and physical environments matching cognitive expectations

### Co-Evolution of Model and Niche

Niche construction creates a feedback loop:

```
Organism's model -> shapes environment -> shapes selection pressures -> shapes model
```

This can lead to:
- **Niche conformance**: Model adapts to niche (classical adaptation)
- **Niche construction**: Niche adapts to model (extended phenotype)
- **Co-adaptation**: Model and niche co-evolve (gene-culture co-evolution)

Under the FEP, niche construction is not an anomaly but an expected consequence of active inference extended across generations.

## The Generative Model as Phenotype

### Genetic Encoding of Priors

The genome encodes the organism's generative model:

```
Genome -> Development -> Generative model (phenotype)
         p(s, theta)    p(o | s, theta) and p(s | theta)
```

Specifically:
- **Morphology** encodes the observation model: body shape determines what observations are possible
- **Neural architecture** encodes the inference algorithm: brain structure determines how inference is performed
- **Innate behaviors** encode prior preferences: reflexes and drives implement homeostatic priors
- **Developmental programs** encode the learning algorithm: how the model is updated during ontogenesis

### Evolution as Bayesian Model Reduction

Across generations, evolution performs a form of **Bayesian model reduction**: eliminating phenotypic features (model parameters) that do not contribute to model evidence (fitness).

```
If a trait does not improve model evidence: p(o | m_with_trait) <= p(o | m_without_trait)
Then selection pressure removes the trait (reduces complexity without losing accuracy)
```

This is the evolutionary analog of synaptic pruning: unnecessary features are eliminated because they increase complexity without improving accuracy.

## Baldwin Effect and Genetic Assimilation

### Learning Guiding Evolution

The Baldwin effect -- where learned behaviors influence the direction of evolution -- has a natural FEP interpretation:

1. **Learning** (within lifetime): Organism learns parameter theta* that minimizes free energy
2. **Fitness advantage**: Organisms that can learn theta* have higher model evidence
3. **Selection**: Genes that make theta* easier to learn (or innately encode it) are selected for
4. **Genetic assimilation**: Eventually, theta* becomes genetically encoded (the prior shifts toward the learned optimum)

In FEP terms, the learned posterior becomes the innate prior across generations:

```
Generation 1: p(theta) = prior; q(theta | data) = posterior (learned)
Generation n: p(theta) -> q(theta | data) from previous generations
```

The prior converges on the posterior -- what was learned becomes innate.

## Evolutionary Transitions and Nested Blankets

### Major Evolutionary Transitions

Major evolutionary transitions (Maynard Smith & Szathmary, 1995) can be understood as the formation of new Markov blankets:

| Transition | New Blanket | Internal States | External States |
|-----------|-------------|-----------------|-----------------|
| Molecules -> Protocells | Cell membrane | Chemistry | Environment |
| Prokaryotes -> Eukaryotes | Nuclear membrane | Nucleus | Cytoplasm |
| Unicellular -> Multicellular | Body boundary | Cells | Environment |
| Individuals -> Colonies | Colony boundary | Individuals | Ecosystem |
| Primates -> Societies | Cultural boundary | Individuals | Other groups |

Each transition creates a new level of organization with its own Markov blanket, enabling free energy minimization at a new scale.

### The Emergence of New Agents

Each major transition creates a new "agent" -- a system with a Markov blanket that minimizes free energy at its own level:

```
Before: Individual cells minimize free energy independently
After: Multicellular organism minimizes free energy as a whole
       (cells' free energy minimization is subordinated to organism-level minimization)
```

This is formalized through nested Markov blankets: the organism's blanket encompasses the blankets of its constituent cells.

## Key References

1. Campbell, J. O. (2016). Universal Darwinism as a process of Bayesian inference. *Frontiers in Systems Neuroscience*, 10, 49.
2. Kauffman, S. A. (1993). *The Origins of Order: Self-Organization and Selection in Evolution*. Oxford University Press.
3. Laland, K. N., et al. (2015). The extended evolutionary synthesis: its structure, assumptions and predictions. *Proceedings of the Royal Society B*, 282(1813), 20151019.
4. Ramstead, M. J. D., Badcock, P. B., & Friston, K. J. (2018). Answering Schrodinger's question: A free-energy formulation. *Physics of Life Reviews*, 24, 1-16.
5. Friston, K. (2013). Life as we know it. *Journal of the Royal Society Interface*, 10(86), 20130475.
6. Maynard Smith, J., & Szathmary, E. (1995). *The Major Transitions in Evolution*. Oxford University Press.
