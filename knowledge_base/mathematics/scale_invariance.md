---
title: Scale Invariance
description: "Invariance of laws or statistics under rescaling, from power laws and self-similarity to renormalisation, and the nested Markov blanket structure that free energy accounts of hierarchy rely on."
type: mathematical_concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - scaling
  - power_laws
  - hierarchy
semantic_relations:
  - type: relates
    links:
      - "[[renormalization_group]]"
      - "[[hierarchical_processing]]"
      - "[[phase_transitions]]"
      - "[[markov_blanket]]"
---

# Scale Invariance

## Overview
A system or function is scale invariant if rescaling its arguments leaves its form unchanged, up to a factor. Formally, $f(\lambda x)=\lambda^{\Delta}f(x)$ for all $\lambda>0$: $f$ is homogeneous of degree $\Delta$. Power laws $f(x)=x^{-\alpha}$ satisfy this with $\Delta=-\alpha$; exponentials and Gaussians do not, because they carry a characteristic scale.

## Formalism
**Power laws and self-similarity.** A distribution $p(x)\propto x^{-\alpha}$ has no preferred scale, and its ratio $p(\lambda x)/p(x)=\lambda^{-\alpha}$ depends only on $\lambda$. Fractals are self-similar under discrete rescalings, giving log-periodic corrections.

**Criticality.** At a continuous phase transition correlation length diverges and fluctuations appear on all scales, so observables follow power laws with universal exponents ([[phase_transitions]]).

**Renormalisation.** The [[renormalization_group]] studies how a description changes under coarse-graining. Fixed points of the flow are scale invariant, and relevant versus irrelevant couplings explain universality. Scale invariance is typically an approximate property over a finite range of scales, bounded by system size and microscopic cutoffs.

**Related notions.** Conformal invariance is a stronger local rescaling symmetry in some critical systems; dimensionless ratios in dimensional analysis express scale covariance of physical law.

## Role in Active Inference
Two uses should be separated. Empirically, neural recordings show scale-free features such as avalanche size distributions (Beggs and Plenz, 2003) and $1/f$ spectra, and critical-brain proposals link them to optimal information processing; this is an open empirical debate, and Active Inference does not predict it by itself. Structurally, the free energy principle is formulated for nested systems: Markov blankets of Markov blankets, with the same variational form at each level ([[hierarchical_processing]], [[bayesian_renormalization]]). Friston describes this as a renormalisation-group-like recursion, where coarse-graining of blanketed states yields a new blanketed system. That is a formal proposal about how the same free-energy form recurs across levels, not a demonstration that dynamics at all levels follow power laws; and exact scale invariance of cortical hierarchies is not assumed.

## Related concepts
- [[renormalization_group]] — the formal machinery for coarse-graining
- [[bayesian_renormalization]] — renormalisation applied to inference
- [[complex_systems]] — scaling in self-organising systems
- [[hierarchical_models]] — levels with analogous structure
- [[markov_blanket]] — the boundary that nests across scales

## References
- Beggs, J. M. and Plenz, D. (2003). Neuronal avalanches in neocortical circuits. Journal of Neuroscience.
