---
title: Fokker-Planck Equation
type: concept
status: active
created: 2025-02-06
updated: 2025-02-06
tags:
  - fokker-planck
  - stochastic-dynamics
  - probability-density
  - non-equilibrium
  - bayesian-mechanics
  - free-energy-principle
semantic_relations:
  - type: foundation
    links:
      - [[mathematics/stochastic_processes|Stochastic Processes]]
      - [[mathematics/differential_equations|Differential Equations]]
      - [[mathematics/probability_theory|Probability Theory]]
  - type: relates
    links:
      - [[mathematics/langevin_dynamics|Langevin Dynamics]]
      - [[mathematics/non_equilibrium_steady_state|Non-Equilibrium Steady State]]
      - [[cognitive/free_energy_principle|Free Energy Principle]]
  - type: implements
    links:
      - [[mathematics/variational_free_energy|Variational Free Energy]]
      - [[mathematics/markov_blankets|Markov Blankets]]
---

# Fokker-Planck Equation

## Overview

The Fokker-Planck equation (FPE), also known as the Kolmogorov forward equation, describes the time evolution of the probability density function of a stochastic process. It is the deterministic equation governing how the probability distribution over states evolves, even though individual trajectories are stochastic. In the context of the free energy principle (FEP), the Fokker-Planck equation is foundational: it provides the mathematical machinery for describing how the density of states evolves toward (and maintains) a non-equilibrium steady state (NESS), and it formalizes the connection between stochastic dynamics, free energy, and Bayesian mechanics.

## Derivation from Langevin Dynamics

### The Langevin Equation

Consider a system described by a Langevin (stochastic differential) equation:

```math
\dot{x}(t) = f(x, t) + \sigma(x, t) \cdot \omega(t)
```

where:
- `x(t)` is the state vector
- `f(x, t)` is the deterministic drift (force divided by friction)
- `sigma(x, t)` is the noise amplitude (diffusion matrix)
- `omega(t)` is white Gaussian noise with `<omega(t)> = 0` and `<omega(t) omega(t')^T> = delta(t-t') I`

### Derivation of the FPE

The FPE for the probability density `p(x, t)` is derived by considering how probability is transported by the stochastic flow. Starting from the Kramers-Moyal expansion and truncating at second order (valid for Gaussian noise):

```math
\frac{\partial p(x, t)}{\partial t} = -\sum_i \frac{\partial}{\partial x_i}[f_i(x, t) p(x, t)] + \frac{1}{2} \sum_{i,j} \frac{\partial^2}{\partial x_i \partial x_j}[D_{ij}(x, t) p(x, t)]
```

where `D = sigma sigma^T` is the diffusion tensor.

### Compact Notation

In compact (operator) notation:

```math
\frac{\partial p}{\partial t} = -\nabla \cdot (f p) + \frac{1}{2} \nabla \cdot (D \nabla p)
```

or equivalently, using the probability current `J`:

```math
\frac{\partial p}{\partial t} = -\nabla \cdot J
```

where:

```math
J = f p - \frac{1}{2} \nabla \cdot (D p) = f p - \frac{1}{2} D \nabla p - \frac{1}{2} (\nabla \cdot D) p
```

This is a continuity equation for probability: probability is conserved and flows according to the current `J`.

### Ito vs. Stratonovich

The precise form of the FPE depends on the stochastic calculus convention:

**Ito convention** (noise independent of current state):
```math
\frac{\partial p}{\partial t} = -\nabla \cdot (f p) + \frac{1}{2} \nabla \cdot (D \nabla p)
```

**Stratonovich convention** (noise symmetrically correlated with state):
```math
\frac{\partial p}{\partial t} = -\nabla \cdot (\tilde{f} p) + \frac{1}{2} \nabla \cdot (D \nabla p)
```

where `tilde{f}` includes a noise-induced drift correction. The FEP literature typically uses the Stratonovich convention because it preserves the chain rule and has a more natural physical interpretation.

## Forward and Backward Equations

### Forward Equation (Fokker-Planck)

The forward equation propagates the probability density forward in time:

```math
\frac{\partial p(x, t)}{\partial t} = \mathcal{L}^{\dagger} p(x, t)
```

where `L^dagger` is the Fokker-Planck operator (adjoint of the generator):

```math
\mathcal{L}^{\dagger} p = -\nabla \cdot (f p) + \frac{1}{2} \nabla \cdot (D \nabla p)
```

Given an initial distribution `p(x, 0)`, the forward equation determines `p(x, t)` for all `t > 0`.

### Backward Equation (Kolmogorov)

The backward equation describes the expected value of a function of the future state, conditioned on the present state:

```math
-\frac{\partial u(x, t)}{\partial t} = \mathcal{L} u(x, t)
```

where `L` is the generator (infinitesimal operator):

```math
\mathcal{L} u = f \cdot \nabla u + \frac{1}{2} \text{tr}(D \nabla^2 u)
```

The forward and backward operators are adjoint: `<L^dagger p, u> = <p, L u>` with appropriate boundary conditions.

### Relationship

The forward equation tells us how distributions evolve; the backward equation tells us how expectations evolve. Both are essential:
- The forward equation is used to compute the steady-state distribution
- The backward equation is used to compute expected values and first-passage times

## Stationary Solutions

### Steady-State Density

The stationary (steady-state) solution `p_ss(x)` satisfies:

```math
\frac{\partial p_{ss}}{\partial t} = 0 \implies \nabla \cdot J_{ss} = 0
```

This does not mean `J_ss = 0` (that would be detailed balance / equilibrium). At NESS, the probability current is divergence-free but may be nonzero -- probability circulates in closed loops.

### Equilibrium (Detailed Balance) Case

When detailed balance holds, the steady-state current vanishes: `J_ss = 0`. For constant diffusion `D`, the equilibrium density takes the Boltzmann-Gibbs form:

```math
p_{eq}(x) = \frac{1}{Z} \exp\left(-\frac{2}{D} V(x)\right)
```

where `V(x)` is the potential such that `f(x) = -nabla V(x)` (gradient flow) and `Z` is the normalization constant.

### NESS (Non-Detailed Balance) Case

When detailed balance is violated (the drift has a non-gradient component), the steady state is a NESS with nonzero probability currents. The NESS density cannot generally be written in simple closed form but satisfies:

```math
\nabla \cdot [f p_{ss} - \frac{1}{2} D \nabla p_{ss}] = 0
```

Living systems are paradigmatic examples of NESS: they maintain organized states through continuous dissipation, with nonzero probability currents reflecting the irreversible cycles that characterize life.

## Connection to Free Energy

### Free Energy Functional

Given a reference distribution `p_ss(x)` (the NESS density), the free energy of any distribution `p(x, t)` relative to `p_ss` is:

```math
F[p] = D_{KL}[p || p_{ss}] = \int p(x) \ln \frac{p(x)}{p_{ss}(x)} dx \geq 0
```

This is the relative entropy (KL divergence) of the current distribution from the steady state.

### Free Energy Dissipation

Under Fokker-Planck dynamics, the free energy decreases monotonically:

```math
\frac{dF}{dt} = -\int \frac{J^2}{Dp} dx \leq 0
```

where `J` is the probability current. This is a form of the H-theorem: the system relaxes toward its steady state by minimizing free energy.

The rate of free energy dissipation equals the entropy production rate, connecting the FPE to the second law of thermodynamics.

### Variational Free Energy in the FEP

In the FEP, an agent's internal states parametrize an approximate distribution `Q_mu(eta)` over external states. The variational free energy:

```math
F(\mu) = D_{KL}[Q_\mu(\eta) || P(\eta | b)]
```

can be related to the Fokker-Planck framework by recognizing that the dynamics of internal states `mu(t)` trace out a path in the space of distributions `Q_mu`, and this path follows the gradient of `F`.

## Non-Equilibrium Steady States

### NESS and the FEP

The FEP posits that living systems maintain themselves at NESS. The Fokker-Planck equation provides the formal framework for understanding these steady states:

1. The organism's state space includes internal, blanket, and external states
2. The coupled dynamics of these states are described by a multi-dimensional Langevin equation
3. The resulting Fokker-Planck equation has a NESS solution that characterizes the typical states of the organism
4. The organism's tendency to remain near typical states (minimizing surprisal) is equivalent to maintaining the NESS

### Surprisal and NESS Density

The surprisal of a state x under the NESS density:

```math
\mathfrak{S}(x) = -\ln p_{ss}(x)
```

is the quantity that the FEP says organisms minimize (on average). States with high surprisal are atypical -- they are states the organism rarely visits. Maintaining low average surprisal is equivalent to maintaining the NESS.

## Helmholtz Decomposition

### Decomposing the Flow

The central mathematical tool connecting the FPE to the FEP is the Helmholtz decomposition of the flow `f(x)` into gradient (dissipative) and solenoidal (conservative) components:

```math
f(x) = -(D + Q) \nabla \ln p_{ss}(x) + \nabla \cdot D
```

where:
- `D` is the (symmetric) diffusion tensor
- `Q` is an antisymmetric matrix (the solenoidal flow)

This can be rewritten as:

```math
f(x) = \underbrace{-D \nabla \ln p_{ss}(x)}_{\text{dissipative (gradient) flow}} + \underbrace{-Q \nabla \ln p_{ss}(x)}_{\text{solenoidal (curl) flow}} + \underbrace{\nabla \cdot D}_{\text{noise gradient correction}}
```

### Dissipative and Solenoidal Flows

**Dissipative (gradient) flow**: `f_D = -D nabla ln p_ss`
- Moves probability down the free energy gradient
- Drives the system toward the NESS
- Produces entropy
- Analogous to relaxation in statistical physics

**Solenoidal (curl) flow**: `f_Q = -Q nabla ln p_ss`
- Circulates probability around contours of constant density
- Preserves the NESS (does not change `p_ss`)
- Does not produce entropy
- Characterizes the non-equilibrium nature of living systems

### Significance for the FEP

The Helmholtz decomposition is crucial for the FEP because:

1. **Internal states follow gradient flow**: The dissipative component of internal state dynamics follows the gradient of a free energy functional. This is what justifies the interpretation of internal dynamics as approximate Bayesian inference.

2. **Solenoidal flow enables circulation**: Living systems are not simply gradient-descent machines. The solenoidal component allows cyclical dynamics (circadian rhythms, homeostatic cycles, behavioral routines) that maintain the NESS without changing the steady-state density.

3. **The decomposition is unique**: Given `p_ss` and `D`, the Helmholtz decomposition of `f` into `D` and `Q` components is unique, providing a principled way to separate "inference-like" dynamics from "circulation-like" dynamics.

## Relation to Bayesian Mechanics

### The Bayesian Mechanics Program

Bayesian mechanics (Friston, 2019; Sakthivadivel, 2022) uses the Fokker-Planck equation as its starting point. The program proceeds as follows:

1. Start with a coupled stochastic dynamical system (Langevin equation)
2. Identify the Markov blanket (particular partition)
3. Write the Fokker-Planck equation for the joint density
4. Compute the NESS density `p_ss(mu, s, a, eta)`
5. Apply the Helmholtz decomposition to the autonomous state dynamics
6. Show that the dissipative flow of internal states minimizes variational free energy

### The Free Energy Lemma (FPE Perspective)

Starting from the FPE and its NESS, one can prove that the expected flow of internal states minimizes:

```math
F(\mu, b) = \mathbb{E}_{q_\mu(\eta)}[-\ln p(b, \eta)] - H[q_\mu(\eta)]
```

This establishes that the internal dynamics "look like" variational inference, providing the mathematical foundation for the FEP.

### Path Integral Formulation

The FPE can be reformulated as a path integral:

```math
p(x_T | x_0) = \int \mathcal{D}[x] \exp\left(-\int_0^T \mathcal{L}(\dot{x}, x, t) dt\right)
```

where `L` is the Onsager-Machlup Lagrangian. This connects the FPE to the path integral formulations of the FEP and to least-action principles.

## Computational Methods

### Numerical Solution

The FPE can be solved numerically using:

1. **Finite differences**: Discretize space and time, solve the resulting system of ODEs
2. **Finite elements**: Better for irregular geometries and adaptive mesh refinement
3. **Spectral methods**: Expand in eigenfunctions of the Fokker-Planck operator for efficient computation
4. **Monte Carlo**: Sample trajectories from the Langevin equation and estimate the density

### Eigenfunction Expansion

The FPE can be analyzed through its eigenfunction expansion:

```math
p(x, t) = p_{ss}(x) + \sum_{n=1}^{\infty} c_n \phi_n(x) e^{-\lambda_n t}
```

where `phi_n` are eigenfunctions and `lambda_n` are eigenvalues of the Fokker-Planck operator (with `lambda_0 = 0` corresponding to `p_ss`). The smallest nonzero eigenvalue `lambda_1` determines the relaxation rate -- how fast the system approaches steady state.

### Mean First-Passage Times

The backward equation can be used to compute mean first-passage times -- the average time for a stochastic process to reach a target set. This is relevant for the FEP in contexts like:
- Foraging: Mean time to find resources
- Decision-making: Mean deliberation time
- Allostatic regulation: Mean time to restore physiological variables

## Multi-Dimensional FPE

### High-Dimensional Systems

For systems with many degrees of freedom (as in biological systems), the FPE becomes a partial differential equation in high-dimensional space. Direct solution is generally intractable, motivating:

1. **Mean-field approximations**: Assume the joint density factorizes
2. **Moment closure**: Track low-order moments and close the hierarchy
3. **Variational methods**: Approximate `p(x, t)` within a parametric family
4. **Particle methods**: Represent `p` through an ensemble of particles

### FPE on Graphs

For discrete-state systems (as in POMDP-based active inference), the FPE becomes a master equation:

```math
\frac{dp_i}{dt} = \sum_j (W_{ij} p_j - W_{ji} p_i)
```

where `W_ij` is the transition rate from state j to state i. This connects the continuous-state FPE formalism to discrete active inference.

## Key References

- Risken, H. (1989). The Fokker-Planck Equation. Springer.
- Friston, K. (2019). A free energy principle for a particular physics. arXiv:1906.10184.
- Pavliotis, G. A. (2014). Stochastic Processes and Applications. Springer.
- Gardiner, C. W. (2009). Stochastic Methods. Springer.
- Sakthivadivel, D. (2022). Towards a geometry and analysis for Bayesian mechanics. arXiv:2204.11900.
- Da Costa, L., et al. (2021). Bayesian mechanics for stationary processes. Proceedings of the Royal Society A, 477(2256).

## Cross-References

- [[mathematics/langevin_dynamics|Langevin Dynamics]] - The stochastic equation whose density the FPE describes
- [[mathematics/non_equilibrium_steady_state|Non-Equilibrium Steady State]] - Stationary solutions of the FPE
- [[cognitive/free_energy_principle|Free Energy Principle]] - Theoretical framework built on FPE
- [[mathematics/markov_blankets|Markov Blankets]] - Partition structure on which Bayesian mechanics operates
- [[mathematics/stochastic_processes|Stochastic Processes]] - Mathematical foundation
- [[mathematics/variational_free_energy|Variational Free Energy]] - The functional minimized by internal state dynamics
- [[mathematics/path_integral|Path Integral]] - Alternative formulation of stochastic dynamics
- [[systems/circular_causality|Circular Causality]] - Solenoidal flows and cyclic dynamics
