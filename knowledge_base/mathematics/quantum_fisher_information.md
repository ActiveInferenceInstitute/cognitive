---
title: Quantum Fisher Information
description: The quantum generalisation of Fisher information, which bounds how precisely a parameter encoded in a quantum state can be estimated and defines a metric on the space of quantum states.
type: mathematical_concept
status: stable
created: 2026-10-12
updated: 2026-10-12
tags:
  - quantum_information
  - estimation_theory
  - information_geometry
aliases:
  - SLD Fisher information
semantic_relations:
  - type: generalizes
    links:
      - "[[fisher_information]]"
  - type: relates
    links:
      - "[[cramer_rao_bound]]"
      - "[[fisher_rao_metric]]"
      - "[[information_geometry]]"
      - "[[quantum_information]]"
      - "[[quantum_mechanics]]"
---

# Quantum Fisher Information

## Overview
Classical Fisher information measures how much a probability distribution $p_\theta(x)$ tells an observer about the parameter $\theta$ (see [[fisher_information]]). In a quantum setting the parameter is encoded in a density operator $\rho_\theta$, and the amount that can be learned depends on which measurement is performed. The quantum Fisher information (QFI) is the maximum of the classical Fisher information over all measurements, and it is a property of the family of states alone.

## Definition
The most common version uses the symmetric logarithmic derivative (SLD) $L_\theta$, the Hermitian operator defined by

$$\partial_\theta \rho_\theta = \tfrac{1}{2}\left(L_\theta \rho_\theta + \rho_\theta L_\theta\right),$$

and sets

$$F_Q(\theta)=\operatorname{Tr}\!\left[\rho_\theta L_\theta^{2}\right].$$

For a pure state $\rho_\theta=|\psi_\theta\rangle\langle\psi_\theta|$ this reduces to

$$F_Q = 4\left(\langle\partial_\theta\psi|\partial_\theta\psi\rangle-\left|\langle\psi|\partial_\theta\psi\rangle\right|^{2}\right),$$

four times the variance of the generator when $|\psi_\theta\rangle=e^{-i\theta G}|\psi_0\rangle$. If all $\rho_\theta$ commute, the SLD construction gives back the classical Fisher information of the shared eigenvalue distribution.

## Estimation bound
For $\nu$ independent repetitions of the experiment, any unbiased estimator $\hat\theta$ obeys the quantum Cramer-Rao bound

$$\operatorname{Var}(\hat\theta)\;\ge\;\frac{1}{\nu F_Q(\theta)} .$$

This is the standard tool of quantum metrology, where states with large $F_Q$, such as entangled states, are prepared to improve sensitivity; Toth and Apellaniz (2014) review how $F_Q$ is used there and its relation to entanglement.

## Geometry
Braunstein and Caves (1994) showed that $F_Q$ is, up to a factor, the metric that distinguishability of neighbouring states induces on the space of quantum states. For pure states it is the Fubini-Study metric, and for mixed states it is the Bures metric. Unlike the classical case, where the Fisher-Rao metric is the unique monotone Riemannian metric on probability distributions up to scale (Chentsov's theorem), there is a whole family of monotone metrics on quantum states. Petz (1996) characterised them, and the SLD choice is one member of that family. Different choices give different quantum Fisher informations, so the phrase refers to the SLD version unless it is stated otherwise.

## Relation to Active Inference
The connection is a formal analogy. Information-geometric treatments of Active Inference rely on the classical Fisher information and the Fisher-Rao metric (see [[information_geometry]]). If beliefs were represented by density operators, the QFI and the other monotone metrics above would be the corresponding quantities; this is an observation made on this page, not a result taken from a cited source. A quantum formulation of the free energy principle exists (Fields et al., 2022), but this page does not claim that it is expressed through the QFI, and the standard Active Inference literature does not depend on it.

## Related concepts
- [[fisher_information]] — the classical quantity this generalises
- [[cramer_rao_bound]] — the classical precision limit
- [[fisher_rao_metric]] — the classical statistical metric
- [[quantum_information]] — density operators and entropies
- [[information_geometry]] — geometry of families of states

## References
- Braunstein, S. L., & Caves, C. M. (1994). Statistical distance and the geometry of quantum states. Physical Review Letters, 72(22), 3439-3443. https://doi.org/10.1103/PhysRevLett.72.3439
- Petz, D. (1996). Monotone metrics on matrix spaces. Linear Algebra and its Applications, 244, 81-96. https://doi.org/10.1016/0024-3795(94)00211-8
- Toth, G., & Apellaniz, I. (2014). Quantum metrology from a quantum information science perspective. Journal of Physics A: Mathematical and Theoretical, 47(42), 424006. https://doi.org/10.1088/1751-8113/47/42/424006
- Fields, C., Friston, K., Glazebrook, J. F., & Levin, M. (2022). A free energy principle for generic quantum systems. Progress in Biophysics and Molecular Biology, 173, 36-59. https://doi.org/10.1016/j.pbiomolbio.2022.05.006
