---
title: "Bayesian Mechanics for Stationary Processes"
authors:
  - "Lancelot Da Costa"
  - "Karl J. Friston"
  - "Conor Heins"
  - "Grigorios A. Pavliotis"
type: citation
status: verified
created: 2025-01-01
year: 2021
journal: "Proceedings of the Royal Society A"
volume: 477
issue: 2256
pages: 20210518
doi: "10.1098/rspa.2021.0518"
tags:
  - bayesian_mechanics
  - stationary_processes
  - markov_blankets
  - physics
  - mathematical
semantic_relations:
  - type: foundational_for
    links:
      - bayesian mechanics
      - stationary processes
  - type: extends
    links:
      - [[friston_2019_particular]]
      - [[kirchhoff_2018]]
  - type: cited_by
    links:
      - [[parr_pezzulo_friston_2022]]
---

# Bayesian Mechanics for Stationary Processes

## Authors
- **Lancelot Da Costa** (Imperial College London)
- **Karl J. Friston** (UCL)
- **Conor Heins** (UCL)
- **Grigorios A. Pavliotis** (Imperial College London)

## Publication Details
- **Journal**: Proceedings of the Royal Society A
- **Year**: 2021
- **Volume**: 477
- **Issue**: 2256
- **Pages**: 20210518
- **DOI**: [10.1098/rspa.2021.0518](https://doi.org/10.1098/rspa.2021.0518)

## Abstract
This paper provides rigorous mathematical foundations for Bayesian mechanics -- the physics of systems that possess Markov blankets. Building on Friston's "particular physics," the authors formalize the conditions under which stationary processes with Markov blankets can be said to perform Bayesian inference. They prove key results about the relationship between the stationary density of a random dynamical system and the variational free energy minimized by its internal states, establishing Bayesian mechanics as a mathematically well-defined framework.

## Key Contributions

### Rigorous Foundations
- **Mathematical Proofs**: Formal proofs of key claims in Bayesian mechanics
- **Conditions Specified**: Precise conditions for Bayesian mechanics to apply
- **Existence Results**: When Markov blankets exist in random dynamical systems
- **Convergence**: When internal dynamics converge to free energy minimizers

### Stationary Process Analysis
- **Nonequilibrium Steady State**: Characterization of NESS with Markov blankets
- **Fokker-Planck**: Connection to Fokker-Planck equation and stationary densities
- **Detailed Balance**: When detailed balance holds and when it does not
- **Solenoidal Flow**: Role of solenoidal (non-gradient) flow

### Key Theorems
- **Free Energy Lemma**: Internal states minimize variational free energy at NESS
- **Particular Partition Theorem**: Conditions for the existence of particular partitions
- **Inference Theorem**: When internal dynamics can be described as inference
- **Consistency**: Self-consistency of the Bayesian mechanics framework

### Connections to Existing Mathematics
- **Stochastic Processes**: Connection to theory of Markov processes
- **Variational Analysis**: Relationship to calculus of variations
- **Information Geometry**: Geometric structure of inference
- **Statistical Mechanics**: Parallels with equilibrium and nonequilibrium stat mech

## Core Concepts

### Stationary Processes with Markov Blankets
Consider a Langevin equation:
```
dx = f(x)dt + sigma dW
```

At the nonequilibrium steady state:
```
L* p_ss(x) = 0   (Fokker-Planck equation, stationary solution)
```

If the state space admits a particular partition `x = {eta, s, a, mu}` with conditional independence structure (Markov blanket), then internal states `mu` can be shown to minimize variational free energy.

### The Free Energy Lemma (Rigorous)
Under specific conditions (smooth dynamics, ergodicity, Markov blanket structure):
```
f_mu(b, mu) = (Q_mu - Gamma_mu) * nabla_mu F(s, mu) + epsilon
```

Where:
- `f_mu`: Drift of internal states
- `Q_mu`: Solenoidal (antisymmetric) component
- `Gamma_mu`: Dissipative (symmetric) component
- `F`: Variational free energy
- `epsilon`: Higher-order terms (vanish under Laplace approximation)

### Conditions for Bayesian Mechanics
The framework applies when:
1. The system is at or near a nonequilibrium steady state
2. A particular partition (Markov blanket) exists
3. The dynamics are smooth and ergodic
4. The noise structure is compatible with the partition

### What Bayesian Mechanics Does NOT Claim
- Does NOT require consciousness or cognition
- Does NOT claim all systems with Markov blankets are "intelligent"
- Does NOT violate physics or require new forces
- Does NOT conflate mathematical description with physical mechanism

## Mathematical Formalism

### Langevin Dynamics at NESS
```
dx = f(x)dt + sqrt(2Gamma) dW
f(x) = (Q(x) - Gamma) * nabla ln p_ss(x)
```

Where:
- `Q`: State-dependent antisymmetric (solenoidal) matrix
- `Gamma`: Diffusion matrix (symmetric, positive semidefinite)
- `p_ss`: Stationary density

### Variational Free Energy
For internal states given blanket states:
```
F(s, mu) = E_{q(eta|mu)}[ln q(eta|mu) - ln p(s, eta)]
```

Under Laplace approximation:
```
F(s, mu) approx -ln p(s, eta*(mu)) + 1/2 ln |H|
```

Where `eta*(mu)` is the mode of the conditional density.

### Fokker-Planck and Markov Blankets
The stationary Fokker-Planck equation:
```
0 = -nabla . (f * p_ss) + nabla . (Gamma * nabla p_ss)
```

The Markov blanket structure imposes constraints on the drift `f` and diffusion `Gamma`.

## Impact and Applications

### Mathematics
- **New Field**: Establishes Bayesian mechanics as a mathematically rigorous discipline
- **Existence Proofs**: Formal conditions for key claims
- **Open Problems**: Identifies key mathematical questions for future work

### Physics
- **Nonequilibrium Physics**: New perspective on NESS
- **Self-Organization**: Mathematical foundation for self-organizing systems
- **Information Physics**: Bridge between information theory and physics

### Biology
- **Theoretical Biology**: Rigorous foundation for FEP in biology
- **Systems Biology**: Mathematical tools for analyzing biological systems
- **Biophysics**: Interface of physics, information, and life

### Philosophy
- **Naturalism**: Grounds mental properties in physical dynamics
- **Emergence**: Formal account of how inference emerges from physics
- **Dual-Aspect Monism**: Mathematical basis for dual descriptions

## Related Work

### Foundational Papers
- [[friston_2019_particular]] - A free energy principle for a particular physics
- [[kirchhoff_2018]] - Markov blankets of life

### Mathematical Context
- [[buckley_2017]] - Mathematical review of continuous FEP

### Applications
- [[friston_2013]] - Life as we know it
- [[ramstead_2018]] - Variational ecology

### Critiques
- [[andrews_2021]] - Philosophical critique of the FEP

## Citations and Influence
This paper has been central to establishing Bayesian mechanics as a rigorous mathematical framework. It moved the discussion from informal arguments to formal proofs and identified precisely what can and cannot be claimed about systems with Markov blankets. It is essential reading for anyone interested in the mathematical foundations of the free energy principle.

## Reading Guide
1. **Introduction**: Motivation for mathematical rigor
2. **Prerequisites**: Stochastic processes and Fokker-Planck
3. **Particular Partitions**: Formal definition and existence
4. **Free Energy Lemma**: Proof and conditions
5. **Discussion**: What the results mean and do not mean

---

> **Mathematical Rigor**: Provides the formal proofs and precise conditions that the FEP literature previously lacked.

---

> **Bayesian Mechanics**: Establishes Bayesian mechanics as a well-defined mathematical framework with clear assumptions.

---

> **Honest Boundaries**: Carefully delineates what Bayesian mechanics does and does not claim about self-organizing systems.
