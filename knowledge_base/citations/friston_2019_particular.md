---
title: "A Free Energy Principle for a Particular Physics"
authors:
  - "Karl J. Friston"
type: citation
status: verified
created: 2025-01-01
year: 2019
journal: "arXiv"
doi: "10.48550/arXiv.1906.10184"
tags:
  - free_energy
  - bayesian_mechanics
  - particular_partition
  - markov_blankets
  - physics
  - foundational
semantic_relations:
  - type: foundational_for
    links:
      - bayesian mechanics
      - particular partition
  - type: extends
    links:
      - [[friston_2013]]
      - [[kirchhoff_2018]]
      - [[friston_2010]]
  - type: cited_by
    links:
      - [[da_costa_2021_bayesian]]
      - [[parr_pezzulo_friston_2022]]
---

# A Free Energy Principle for a Particular Physics

## Authors
- **Karl J. Friston** (Wellcome Trust Centre for Neuroimaging, UCL)

## Publication Details
- **Journal**: arXiv preprint
- **Year**: 2019
- **DOI**: [10.48550/arXiv.1906.10184](https://doi.org/10.48550/arXiv.1906.10184)

## Abstract
This monograph-length paper provides the most complete and rigorous formulation of the free energy principle as a theory of physics. Friston develops "Bayesian mechanics" -- the physics of systems that possess a Markov blanket -- by starting from the Langevin dynamics of random dynamical systems and showing how the existence of a Markov blanket (the "particular partition") entails that internal states can be described as performing variational inference about external states. This is the most technically ambitious statement of the FEP, extending it from a principle of brain function to a principle of physics.

## Key Contributions

### Bayesian Mechanics
- **New Branch of Physics**: Physics of systems with Markov blankets
- **From Physics to Inference**: How physical dynamics entail inference
- **Particular Partition**: The partition into internal, external, sensory, active states
- **Dual Aspect**: Internal dynamics have both physical and inferential descriptions

### Rigorous Foundation
- **Langevin Dynamics**: Starts from stochastic differential equations
- **Nonequilibrium Steady State**: Systems at their attracting set
- **Solenoidal Flow**: Distinguishes gradient from solenoidal components
- **Fokker-Planck**: Connection to probability density dynamics

### Particular Partition
- **Definition**: A partition of states such that internal and external are conditionally independent given blanket states
- **Existence**: When does a system possess a Markov blanket
- **Dynamics**: How the partition constrains dynamics
- **Ontology**: What exists are things with Markov blankets

### From Physics to Mind
- **Sentient Behavior**: Systems with deep temporal models appear sentient
- **Simple to Complex**: Continuum from particles to organisms
- **Self-Evidencing**: All blanket-possessing things "self-evidence"
- **No Vitalism**: Mind arises from physical dynamics, no special ingredient

## Core Concepts

### The Particular Partition
Any random dynamical system at nonequilibrium steady state (NESS) can be partitioned:
```
x = {eta, s, a, mu}
```

Where:
- `eta`: External states (hidden from the system)
- `s`: Sensory states (influenced by external, influence internal)
- `a`: Active states (influenced by internal, influence external)
- `mu`: Internal states (conditionally independent of external given blanket)
- Blanket `b = {s, a}`

### Free Energy Lemma
If a system has a Markov blanket and is at NESS, then internal states can be described as minimizing variational free energy:

```
f_mu(b, mu) = (Q_mu - Gamma_mu) * nabla_mu F(b, mu)
```

Where:
- `f_mu`: Flow of internal states
- `Q_mu`: Solenoidal flow (dissipative-free)
- `Gamma_mu`: Dissipative flow
- `F`: Variational free energy

### Bayesian Mechanics
The internal dynamics have a dual aspect:
1. **Physical**: They follow Langevin dynamics (stochastic differential equations)
2. **Inferential**: They can be described as performing variational Bayesian inference about external states

This duality is not a metaphor -- it is a mathematical identity.

### Solenoidal vs Gradient Flow
The dynamics decompose into:
```
dx/dt = (Q - Gamma) * nabla ln p(x) + w
```

Where:
- `Q`: Antisymmetric (solenoidal) -- conservative, non-dissipative
- `Gamma`: Symmetric (gradient) -- dissipative
- `w`: Random fluctuations

## Mathematical Formalism

### Langevin Dynamics
The starting point is a stochastic differential equation:
```
dx = f(x)dt + sigma * dW
```

At the nonequilibrium steady state:
```
f(x) = (Q - Gamma) * nabla ln p_ss(x)
```

Where `p_ss` is the stationary density.

### Variational Free Energy
For internal states:
```
F(s, mu) = E_q[ln q(eta|mu) - ln p(s, eta)]
```

Where:
- `q(eta|mu)`: The variational density parameterized by internal states
- `p(s, eta)`: The joint density over sensory and external states

### Self-Evidencing
Systems appear to maximize their model evidence:
```
ln p(s|m) >= -F(s, mu)
```

## Impact and Applications

### Physics
- **Bayesian Mechanics**: New framework for physics of self-organizing systems
- **Nonequilibrium Thermodynamics**: Connection to dissipative structures
- **Statistical Mechanics**: Information-theoretic perspective on NESS

### Philosophy
- **Panpsychism Debate**: Does every Markov blanket-possessing thing "infer"?
- **Ontology**: What exists are things with Markov blankets
- **Mind-Body Problem**: Dual-aspect monism through Bayesian mechanics

### Biology
- **Definition of Life**: Living things are complex Markov blankets
- **Evolution**: From simple to complex inference through evolution
- **Consciousness**: Deep temporal models as substrate for awareness

## Related Work

### Foundational Papers
- [[friston_2013]] - Life as we know it
- [[kirchhoff_2018]] - Markov blankets of life
- [[friston_2010]] - Free energy principle review

### Extensions
- [[da_costa_2021_bayesian]] - Bayesian mechanics for stationary processes
- [[ramstead_2018]] - Variational ecology

### Critiques
- [[andrews_2021]] - Philosophical critique of the FEP

## Citations and Influence
This monograph represents the most ambitious and technically demanding statement of the free energy principle. It has spawned the field of "Bayesian mechanics" and generated intense debate about whether the FEP is a principle of physics, biology, or both. It remains the most complete formal statement of the theory.

## Reading Guide
1. **Prologue**: Motivation and overview
2. **Random Dynamical Systems**: Mathematical prerequisites
3. **The Particular Partition**: Markov blankets in physics
4. **Bayesian Mechanics**: From physics to inference
5. **Self-Organization**: From particles to organisms
6. **Sentience**: Deep temporal models and mind

---

> **Bayesian Mechanics**: Establishes a new branch of physics for systems with Markov blankets, extending the FEP to fundamental physics.

---

> **Particular Partition**: Provides the rigorous mathematical foundation for Markov blankets in random dynamical systems.

---

> **Most Ambitious Statement**: The most complete and technically demanding formulation of the free energy principle.
