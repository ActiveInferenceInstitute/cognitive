---
title: Langevin Dynamics
type: concept
status: active
created: 2025-02-06
updated: 2025-02-06
tags:
  - langevin-dynamics
  - stochastic-differential-equations
  - brownian-motion
  - gradient-flow
  - free-energy-principle
  - sampling
semantic_relations:
  - type: foundation
    links:
      - [[mathematics/stochastic_processes|Stochastic Processes]]
      - [[mathematics/differential_equations|Differential Equations]]
      - [[mathematics/probability_theory|Probability Theory]]
  - type: relates
    links:
      - [[mathematics/fokker_planck|Fokker-Planck Equation]]
      - [[mathematics/non_equilibrium_steady_state|Non-Equilibrium Steady State]]
      - [[cognitive/free_energy_principle|Free Energy Principle]]
  - type: implements
    links:
      - [[mathematics/variational_free_energy|Variational Free Energy]]
      - [[cognitive/active_inference|Active Inference]]
---

# Langevin Dynamics

## Overview

Langevin dynamics describes the evolution of a system subject to both deterministic forces and random fluctuations. Originally formulated by Paul Langevin (1908) to describe Brownian motion, this framework has become central to the free energy principle (FEP) because it provides the mathematical language for describing how biological systems evolve stochastically while maintaining organized states. Under the FEP, the dynamics of any self-organizing system -- from a single cell to a social group -- can be written as a Langevin equation, with the deterministic drift encoding free energy minimization and the stochastic term encoding the irreducible noise of the physical world.

## Stochastic Differential Equations

### The Langevin Equation

The general Langevin equation is:

```math
dx(t) = f(x, t) \, dt + \sigma(x, t) \, dW(t)
```

or equivalently in physicists' notation:

```math
\dot{x}(t) = f(x, t) + \sigma(x, t) \, \xi(t)
```

where:
- `x(t)` is the state vector in R^n
- `f(x, t): R^n x R -> R^n` is the drift (deterministic force)
- `sigma(x, t): R^n x R -> R^{n x m}` is the diffusion (noise amplitude) matrix
- `W(t)` is an m-dimensional Wiener process (standard Brownian motion)
- `xi(t) = dW/dt` is white Gaussian noise (formal derivative of W)

### Properties of White Noise

White noise `xi(t)` has the following statistical properties:

```math
\langle \xi_i(t) \rangle = 0 \quad \text{(zero mean)}
```
```math
\langle \xi_i(t) \xi_j(t') \rangle = \delta_{ij} \delta(t - t') \quad \text{(delta-correlated, independent components)}
```

These properties make `xi(t)` mathematically pathological (it is not a function in the classical sense), necessitating the rigorous framework of stochastic calculus.

### Ito vs. Stratonovich Interpretation

When the noise amplitude `sigma` depends on the state `x`, the Langevin equation is ambiguous -- it must be supplemented with an interpretation:

**Ito interpretation**: The noise is evaluated at the beginning of each time step:
```math
x(t + dt) = x(t) + f(x(t)) dt + \sigma(x(t)) \Delta W(t)
```

**Stratonovich interpretation**: The noise is evaluated at the midpoint:
```math
x(t + dt) = x(t) + f(x(t + dt/2)) dt + \sigma(x(t + dt/2)) \Delta W(t)
```

The two interpretations are related by a noise-induced drift correction:

```math
f_{\text{Strat}} = f_{\text{Ito}} + \frac{1}{2} \sum_j \sigma_{kj} \frac{\partial \sigma_{ij}}{\partial x_k}
```

The FEP literature generally uses the Stratonovich interpretation because it preserves the standard chain rule and has a more natural physical interpretation (the noise represents a physical process, not a mathematical artifact).

## Brownian Motion

### Mathematical Brownian Motion (Wiener Process)

The Wiener process `W(t)` is the mathematical idealization of Brownian motion. Its properties:

1. `W(0) = 0`
2. `W(t)` has independent increments: for `t_1 < t_2 < t_3 < t_4`, `W(t_4) - W(t_3)` is independent of `W(t_2) - W(t_1)`
3. `W(t) - W(s) ~ N(0, t - s)` for `s < t`
4. `W(t)` is continuous in t (with probability 1)
5. `W(t)` is nowhere differentiable (with probability 1)

### Physical Brownian Motion

Physical Brownian motion (the random motion of particles in a fluid) is described by the Langevin equation:

```math
m\ddot{x} = -\gamma \dot{x} + F_{ext}(x) + \sqrt{2\gamma k_B T} \, \xi(t)
```

where `m` is mass, `gamma` is friction, `F_ext` is external force, `k_B` is Boltzmann's constant, and `T` is temperature. In the overdamped limit (inertia negligible), this reduces to:

```math
\gamma \dot{x} = F_{ext}(x) + \sqrt{2\gamma k_B T} \, \xi(t)
```

or, dividing by `gamma`:

```math
\dot{x} = -\nabla V(x) + \sqrt{2D} \, \xi(t)
```

where `V(x)` is the potential energy and `D = k_B T / gamma` is the diffusion coefficient.

### Brownian Motion in the FEP

In the FEP, Brownian motion represents the irreducible stochasticity of the physical world. Even a perfectly adapted organism cannot eliminate noise -- it can only minimize its impact through accurate prediction (free energy minimization). The noise term in the Langevin equation represents:
- Thermal fluctuations in molecular processes
- Ion channel noise in neural signaling
- Environmental unpredictability
- Measurement noise in sensory systems

## Drift and Diffusion

### The Drift Term

The drift `f(x)` determines the deterministic tendency of the system. In the FEP, the drift encodes the system's tendency to move toward states with low surprisal:

```math
f(x) = -(D + Q) \nabla \mathfrak{S}(x) + \nabla \cdot D
```

where `S(x) = -ln p_ss(x)` is the surprisal (self-information) under the steady-state density, `D` is the diffusion tensor, and `Q` is the antisymmetric solenoidal matrix.

### The Diffusion Term

The diffusion tensor `D = sigma sigma^T / 2` determines the amplitude and correlation structure of fluctuations. In the FEP:

- `D` is typically assumed to be state-independent (additive noise) for simplicity
- State-dependent diffusion (multiplicative noise) introduces additional drift terms
- The magnitude of D determines the "temperature" of the system -- higher diffusion means more exploration of the state space

### Drift-Diffusion Balance

The fluctuation-dissipation theorem (see below) ensures that drift and diffusion are balanced at equilibrium. In the FEP, this balance is what allows the system to maintain a steady state: the drift drives the system toward typical (low-surprisal) states, while diffusion perturbs it away, and the balance between these two determines the steady-state density.

## Relation to Gradient Flows

### Gradient Descent in Noise

In the absence of noise, the dynamics reduce to a gradient flow (gradient descent):

```math
\dot{x} = -\nabla V(x) \quad \text{(deterministic gradient flow)}
```

The Langevin equation adds noise to this gradient flow:

```math
\dot{x} = -\nabla V(x) + \sqrt{2D} \, \xi(t)
```

This can be understood as gradient descent on a free energy landscape, where the noise prevents the system from getting trapped in local minima and ensures exploration of the state space.

### Free Energy as Potential

In the FEP, the potential `V(x)` is identified with the surprisal (or free energy):

```math
V(x) \equiv \mathfrak{S}(x) = -\ln p_{ss}(x)
```

The deterministic component of the flow moves the system downhill on the surprisal landscape. States with low surprisal (high probability under the steady-state density) are the attractors of the deterministic dynamics.

### Beyond Gradient Flow: Solenoidal Dynamics

Not all drift is gradient descent. The Helmholtz decomposition separates the drift into gradient and solenoidal components:

```math
f(x) = \underbrace{-D \nabla V(x)}_{\text{gradient (dissipative)}} + \underbrace{Q(x) \nabla V(x)}_{\text{solenoidal (conservative)}} + \underbrace{\nabla \cdot D}_{\text{noise correction}}
```

The solenoidal component circulates the system around contours of constant surprisal without changing the probability density. This is crucial for living systems, which exhibit cyclical dynamics (circadian rhythms, metabolic cycles, behavioral routines) in addition to relaxation toward steady states.

## Fluctuation-Dissipation Theorem

### Statement

The fluctuation-dissipation theorem (FDT) relates the amplitude of spontaneous fluctuations to the system's response to external perturbations:

```math
\langle x(t) x(0) \rangle = \frac{k_B T}{\gamma} e^{-\gamma t / m} \quad \text{(for a simple harmonic oscillator in a bath)}
```

More generally, the FDT states that the noise power spectrum is proportional to the imaginary part of the response function:

```math
S(\omega) = \frac{2 k_B T}{\omega} \text{Im}[\chi(\omega)]
```

### FDT in the FEP

The FDT has a deep connection to the FEP: it ensures that systems at equilibrium cannot gain information from their fluctuations (there is no "free lunch" of information). In non-equilibrium systems (like living organisms), the FDT is violated -- and this violation is precisely what allows organisms to extract information from their environment and maintain organized states far from equilibrium.

The degree of FDT violation can be quantified by the entropy production rate:

```math
\dot{S}_{prod} = \frac{1}{T} \int J(x) \cdot D^{-1} J(x) \frac{dx}{p_{ss}(x)} \geq 0
```

where `J(x)` is the probability current. Living systems have positive entropy production, reflecting their fundamentally non-equilibrium nature.

## Detailed Balance

### Definition

A stochastic system satisfies detailed balance if the probability of any forward transition equals the probability of the reverse transition:

```math
p_{ss}(x) T(x \to y) = p_{ss}(y) T(y \to x) \quad \forall x, y
```

equivalently, the probability current vanishes at steady state: `J_ss(x) = 0` for all x.

### Detailed Balance in Langevin Dynamics

For Langevin dynamics with drift `f(x)` and constant diffusion `D`, detailed balance holds if and only if the drift is a pure gradient:

```math
f(x) = -D \nabla V(x)
```

with no solenoidal component. In this case, the steady-state density is Boltzmann-Gibbs:

```math
p_{ss}(x) \propto \exp(-V(x) / D)
```

### Detailed Balance Violation in Living Systems

Living systems violate detailed balance: they have nonzero probability currents at steady state. This is the hallmark of non-equilibrium systems and is necessary for:
- Sensing (breaking symmetry between stimulus presence and absence)
- Acting (breaking symmetry between doing and not doing)
- Maintaining homeostasis (breaking symmetry between living and dead states)

## Connection to FEP Dynamics

### The Particular Physics

Friston's "particular physics" (2019) starts from the Langevin equation for a system with a Markov blanket:

```math
\dot{\mu} = f_\mu(\mu, b) + \omega_\mu \quad \text{(internal states)}
```
```math
\dot{b} = f_b(\mu, b, \eta) + \omega_b \quad \text{(blanket states)}
```
```math
\dot{\eta} = f_\eta(b, \eta) + \omega_\eta \quad \text{(external states)}
```

The FEP then shows that, at NESS, the autonomous flow `f_mu` can be decomposed to reveal a free energy gradient component.

### Active Inference as Langevin Dynamics

Active inference can be formulated as coupled Langevin equations:

**Perceptual inference** (updating beliefs):
```math
\dot{\mu} = -\kappa_\mu \nabla_\mu F(\mu, o) + \omega_\mu
```

**Active inference** (updating actions):
```math
\dot{a} = -\kappa_a \nabla_a F(\mu, o(a)) + \omega_a
```

Both perception and action are gradient descent on free energy, with noise terms representing neural and motor stochasticity.

### Parameter Learning as Slow Langevin Dynamics

Model parameter learning can be viewed as slow Langevin dynamics on the parameter space:

```math
\dot{\theta} = -\eta \nabla_\theta F(\theta, \mu, o) + \sqrt{2\eta T_{\theta}} \, \xi_\theta(t)
```

where `eta` is the learning rate and `T_theta` is an effective temperature controlling exploration in parameter space. This connects learning to simulated annealing: early in learning, high temperature promotes exploration; later, temperature decreases for exploitation.

## Sampling Methods

### Langevin Monte Carlo

The Langevin equation naturally generates samples from the target distribution `p(x) propto exp(-V(x))`. The Unadjusted Langevin Algorithm (ULA) uses discrete-time updates:

```math
x_{n+1} = x_n - \epsilon \nabla V(x_n) + \sqrt{2\epsilon} \, z_n, \quad z_n \sim N(0, I)
```

For small step size `epsilon`, the samples approximate the target distribution. The Metropolis-Adjusted Langevin Algorithm (MALA) adds a Metropolis-Hastings acceptance step to correct for discretization error.

### Connections to MCMC

Langevin dynamics provides a principled basis for Markov chain Monte Carlo (MCMC) methods:

1. **Langevin MCMC**: Directly discretizes the Langevin equation
2. **Hamiltonian Monte Carlo (HMC)**: Adds momentum variables to create Hamiltonian dynamics, improving mixing
3. **Stochastic gradient Langevin dynamics (SGLD)**: Uses noisy gradient estimates, enabling scalable Bayesian inference

### Sampling in the FEP

The connection between Langevin dynamics and sampling suggests that neural dynamics may implement a form of sampling: the stochastic fluctuations of neural activity explore the posterior distribution over hidden states, with the drift term guiding exploration toward high-probability regions. This "sampling hypothesis" (Fiser et al., 2010) provides a neural implementation of Bayesian inference through stochastic dynamics.

## MCMC Connections

### Langevin as MCMC Kernel

The Langevin equation defines a Markov chain with transition kernel:

```math
K(x_{n+1} | x_n) = N(x_n - \epsilon \nabla V(x_n), 2\epsilon I)
```

This kernel satisfies detailed balance (approximately, for small `epsilon`) with respect to `p(x) propto exp(-V(x))`, ensuring convergence to the target distribution.

### Convergence Rates

The mixing time of Langevin MCMC depends on properties of the target distribution:
- **Strongly log-concave targets**: Mixing in `O(d / epsilon)` steps (d = dimension, epsilon = accuracy)
- **Multi-modal targets**: Exponentially slow mixing due to energy barriers between modes
- **Ill-conditioned targets**: Slow mixing due to disparate scales in different directions (preconditioning helps)

### Relevance to Neural Computation

If the brain implements sampling-based inference via Langevin dynamics, convergence properties determine:
- How quickly the brain can form accurate percepts
- Whether the brain can escape local optima in belief space
- Why neural noise might be functionally important (it enables exploration)
- Why attention (precision) modulation affects perception speed and accuracy

## Applications in the FEP

### Neural Dynamics

Cortical dynamics can be modeled as Langevin equations where:
- Drift = predictive coding update (gradient descent on prediction error)
- Diffusion = neural noise (synaptic, channel, background noise)
- Steady state = the posterior distribution over environmental states

### Behavioral Dynamics

Movement and behavioral trajectories follow Langevin-like dynamics:
- Drift = motor commands derived from active inference
- Diffusion = motor noise and environmental perturbations
- Steady state = the set of adaptive behavioral patterns

### Evolutionary Dynamics

Population-level dynamics in evolutionary biology:
- Drift = selection pressure (gradient of fitness landscape)
- Diffusion = genetic drift (random fluctuations due to finite population)
- Steady state = evolutionary stable strategies

## Key References

- Langevin, P. (1908). Sur la theorie du mouvement brownien. Comptes Rendus de l'Academie des Sciences, 146, 530-533.
- Gardiner, C. W. (2009). Stochastic Methods: A Handbook for the Natural and Social Sciences. Springer.
- Friston, K. (2019). A free energy principle for a particular physics. arXiv:1906.10184.
- Roberts, G. O., & Tweedie, R. L. (1996). Exponential convergence of Langevin distributions and their discrete approximations. Bernoulli, 2(4), 341-363.
- Fiser, J., Berkes, P., Orban, G., & Lengyel, M. (2010). Statistically optimal perception and learning. Trends in Cognitive Sciences, 14(3), 119-130.
- Welling, M., & Teh, Y. W. (2011). Bayesian learning via stochastic gradient Langevin dynamics. ICML.

## Cross-References

- [[mathematics/fokker_planck|Fokker-Planck Equation]] - Density dynamics corresponding to Langevin trajectories
- [[mathematics/non_equilibrium_steady_state|Non-Equilibrium Steady State]] - Stationary states of Langevin systems
- [[cognitive/free_energy_principle|Free Energy Principle]] - Theoretical framework interpreting Langevin dynamics
- [[mathematics/stochastic_processes|Stochastic Processes]] - Mathematical foundation
- [[cognitive/active_inference|Active Inference]] - Action and perception as coupled Langevin flows
- [[mathematics/variational_free_energy|Variational Free Energy]] - The potential for the gradient flow
- [[mathematics/markov_blankets|Markov Blankets]] - Partition structure for coupled Langevin systems
- [[mathematics/path_integral|Path Integral]] - Path integral formulation of Langevin dynamics
