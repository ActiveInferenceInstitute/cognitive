---
title: Critical Periods
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - development
  - plasticity
  - sensory_systems
semantic_relations:
  - type: relates
    links:
      - "[[active_inference]]"
      - "[[neural_plasticity]]"
      - "[[cognitive_development]]"
---

# Critical Periods

## Overview
A critical period is a limited developmental window in which a neural circuit is unusually sensitive to experience, so that appropriate input shapes it permanently and its absence or distortion produces lasting deficits. "Sensitive period" is the usual term for a softer version in which plasticity is high but not exclusive to that window and the effects are less strictly irreversible. The classic example is ocular dominance plasticity in visual cortex.

## Mechanism
Wiesel and Hubel (1963) found that closing one eyelid of a kitten during early life shifted cortical responses toward the open eye and caused lasting loss of acuity in the deprived eye (amblyopia), whereas the same deprivation in adulthood had little effect. Later work identified the following features:

- **Onset** depends on maturation of inhibitory circuitry, in particular parvalbumin-expressing GABAergic interneurons; Hensch (2005) showed that reducing GABA synthesis delays onset and enhancing inhibition can advance it.
- **Closure** involves molecular "brakes" that stabilize circuits, including perineuronal nets of extracellular matrix, myelin-associated signaling through the Nogo receptor, and changes in neuromodulatory signaling.
- **Reopening**: in animal models, removing brakes or altering inhibition can restore juvenile-like plasticity in adults, which motivates therapies for amblyopia.

Other examples cover filial imprinting, birdsong learning, binaural sound localization maps, and, in humans, phonetic category tuning and first-language acquisition, where outcomes depend on age of exposure. For human language, the data support graded sensitive periods more than sharply bounded critical periods.

## Role in Active Inference
This is an area where the free-energy literature offers interpretations rather than established accounts. One reading ties a critical period to the relation between prior precision and learning: an immature system has weak, imprecise priors and high learning rates, so experience dominates; as evidence accumulates, priors sharpen and become costly to revise. Counting-based Dirichlet updates give such a falling learning rate automatically (see [[metaplasticity]]). Another reading connects reduced plasticity to the interplay of excitation and inhibition that sets the precision of prediction errors. These are consistent with the observation that deprivation shifts what the circuit expects, but they have not been tested directly against the molecular brake literature, and a pure counting argument does not explain why plasticity can be pharmacologically reopened, which suggests an actively regulated process.

## Related concepts
- [[neural_plasticity]] — the capacity that critical periods gate
- [[cognitive_development]] — developmental context, including language
- [[pruning]] — experience-dependent elimination of synapses during these windows
- [[stability_plasticity]] — closure trades plasticity for stability
- [[precision_weighting]] — framework term for the weighting of priors against evidence
- [[metaplasticity]] — regulation of plasticity by prior activity

## References
- Wiesel, T. N., & Hubel, D. H. (1963). Single-cell responses in striate cortex of kittens deprived of vision in one eye. *Journal of Neurophysiology*.
- Hensch, T. K. (2005). Critical period plasticity in local cortical circuits. *Nature Reviews Neuroscience*.
