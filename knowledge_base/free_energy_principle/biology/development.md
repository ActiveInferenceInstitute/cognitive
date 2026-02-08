---
title: "Ontogenesis as Hierarchical Inference"
type: concept
status: active
created: 2025-01-01
updated: 2025-02-06
tags:
  - free_energy_principle
  - development
  - morphogenesis
  - critical_periods
  - epigenetics
  - ontogenesis
semantic_relations:
  - type: foundation
    links:
      - [[knowledge_base/free_energy_principle/mathematics/core_principle|Core Principle]]
      - [[knowledge_base/free_energy_principle/mathematics/markov_blankets|Markov Blankets]]
  - type: relates
    links:
      - [[evolution|Evolution]]
      - [[neural_systems|Neural Systems]]
      - [[knowledge_base/free_energy_principle/cognitive/learning|Learning]]
---

# Ontogenesis as Hierarchical Inference

## Overview

Development -- from fertilized egg to mature organism -- can be understood through the FEP as a process of hierarchical inference. The developing organism progressively builds and refines its generative model, starting with the broadest, most abstract structures and filling in increasingly specific details. This maps onto the biological reality of development: gross morphology forms before fine detail, and sensory systems mature from coarse to fine sensitivity.

## Morphogenesis as Free Energy Minimization

### Cellular Self-Organization

Morphogenesis -- the development of form -- is self-organization at the cellular level. Under the FEP, cells are agents with Markov blankets that minimize free energy:

```
Cell:
  Internal states: Gene expression, metabolic state
  Sensory states: Membrane receptors (detect morphogens, cell contacts)
  Active states: Secretion, migration, division, apoptosis
  External states: Other cells, extracellular matrix, morphogen gradients
```

Each cell infers its position and identity from local signals and acts accordingly:

```
q(position, identity) = argmin_q F[q, morphogen_signals]
action = argmin_a G(a) -- migrate, differentiate, divide, or die
```

### Morphogen Gradients as Observations

Morphogen gradients provide the observations that cells use for inference:

```
p(morphogen_concentration | position) -- the generative model
q(position | morphogen_concentration) -- the cell's inference
```

Classic examples:
- **Bicoid gradient** in Drosophila: anterior-posterior axis specification
- **Sonic hedgehog** in vertebrates: ventral-dorsal neural tube patterning
- **Wnt signaling**: multiple axis determinations

The cell "reads" the morphogen gradient and infers its position, then adopts the appropriate fate (gene expression program) for that position.

## Critical Periods and Precision

### Critical Periods as High-Precision Windows

Critical periods -- developmental windows when the brain is maximally plastic -- correspond to periods of high precision on sensory prediction errors:

```
During critical period: Pi_sensory very high
-> Sensory data strongly drives model updating
-> Rapid learning from experience
-> Model parameters change quickly
```

```
After critical period: Pi_sensory reduced (or Pi_prior increased)
-> Prior beliefs dominate
-> Slower learning
-> Model parameters are stable
```

### The Mechanism

Critical period opening and closing is regulated by:
1. **GABA maturation**: Inhibitory circuits mature -> precision becomes better controlled
2. **Perineuronal nets**: Form around parvalbumin interneurons -> stabilize existing circuits
3. **Myelin**: Axon myelination -> locks in temporal dynamics
4. **Neuromodulation**: Developmental changes in ACh, DA, NE levels

Under the FEP, these biological mechanisms implement a transition from high sensory precision (open critical period: learn from data) to high prior precision (closed critical period: rely on learned model).

### Reopening Critical Periods

Critical periods can be experimentally reopened by:
- **Dark rearing** (preventing experience -> keeping priors uninformative)
- **Fluoxetine** (SSRI -> modulates precision weighting)
- **Valproate** (HDAC inhibitor -> epigenetic reset of precision)
- **Environmental enrichment** (increases sensory precision)

All of these interventions can be understood as resetting the precision balance toward sensory data, re-enabling plasticity.

## Hierarchical Development: Coarse to Fine

### The Developmental Sequence

Development proceeds from coarse to fine, from abstract to specific:

```
Stage 1: Body plan (axes, segments)           -- highest-level generative structure
Stage 2: Organ specification (organogenesis)   -- mid-level structure
Stage 3: Tissue differentiation               -- detail within organs
Stage 4: Cellular specialization              -- fine detail
Stage 5: Synaptic refinement (neural)         -- finest detail
```

This mirrors the structure of a hierarchical generative model:
- High levels (coarse, abstract) are specified first -> provide priors for lower levels
- Low levels (fine, specific) are specified later -> refined by experience

### Experience-Expectant vs. Experience-Dependent Development

**Experience-expectant** (Greenough et al., 1987): The generative model expects certain types of experience:
```
p(visual_input) expects: edges, faces, motion
p(auditory_input) expects: speech sounds, environmental sounds
```
The genome encodes the structure of the generative model but expects experience to tune the parameters.

**Experience-dependent**: Unique individual experiences shape the model:
```
Learning a specific language, recognizing specific faces, developing specific skills
```
The parameters are tuned by the individual's actual environment.

## Epigenetics and Generative Model Transmission

### Epigenetic Inheritance as Prior Transfer

Epigenetic marks (DNA methylation, histone modification) can transmit generative model parameters across generations:

```
Parent's experience -> Epigenetic modification -> Offspring's prior
```

This is a form of **intergenerational prior transfer**: the parent's posterior (learned from experience) becomes the offspring's prior (innate tendencies).

Example: Parental stress exposure can alter offspring's stress response through epigenetic modification of HPA axis genes -- the parent's "learned" stress model is partially transmitted as an innate prior in the offspring.

## Neural Development as Model Building

### Synaptogenesis and Pruning

Neural development follows a pattern of overproduction followed by selective elimination:

```
Overproduction: Create many possible connections (large model space)
Activity-dependent pruning: Eliminate connections that don't contribute to model evidence
```

This is biological Bayesian model reduction:
```
If synapse contributes to accuracy: Retained (LTP)
If synapse doesn't contribute: Pruned (LTD, elimination)
Free energy = Complexity - Accuracy
Pruning reduces complexity while (ideally) maintaining accuracy
```

### The Role of Spontaneous Activity

Before sensory experience, the developing nervous system generates **spontaneous activity** (retinal waves, spontaneous motor patterns):

```
Spontaneous activity = self-generated "virtual" observations
```

The developing generative model uses its own activity as training data, bootstrapping the inference machinery before real sensory data arrives. This is analogous to the "wake-sleep" algorithm: the generative model produces samples, and the recognition model learns to infer from them.

## Key References

1. Friston, K., & Buzsaki, G. (2016). The functional anatomy of time: what and when in the brain. *Trends in Cognitive Sciences*, 20(7), 500-511.
2. Levin, M. (2021). Bioelectric signaling: Reprogrammable circuits underlying embryogenesis, regeneration, and cancer. *Cell*, 184(6), 1971-1989.
3. Greenough, W. T., Black, J. E., & Wallace, C. S. (1987). Experience and brain development. *Child Development*, 58(3), 539-559.
4. Hensch, T. K. (2005). Critical period plasticity in local cortical circuits. *Nature Reviews Neuroscience*, 6(11), 877-888.
5. Badcock, P. B., Friston, K. J., & Ramstead, M. J. D. (2019). The hierarchically mechanistic mind: A free-energy formulation of the human psyche. *Physics of Life Reviews*, 31, 104-121.
