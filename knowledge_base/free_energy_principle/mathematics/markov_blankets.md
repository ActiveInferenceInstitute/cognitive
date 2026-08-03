---
title: "Markov Blankets: Statistical Boundaries and Bayesian Mechanics"
type: mathematical_concept
status: active
created: 2025-01-01
updated: 2025-02-06
tags:
  - free_energy_principle
  - markov_blankets
  - particular_partition
  - conditional_independence
  - bayesian_mechanics
  - self_organization
semantic_relations:
  - type: foundation
    links:
      - [[core_principle|Core Principle]]
  - type: relates
    links:
      - [[variational_free_energy|Variational Free Energy]]
      - [[information_geometry|Information Geometry]]
      - [[knowledge_base/free_energy_principle/biology/homeostasis|Homeostasis]]
      - [[knowledge_base/free_energy_principle/systems/self_organization|Self-Organization]]
  - type: extends
    links:
      - [[advanced_formulations|Advanced Formulations]]
---

# Markov Blankets: Statistical Boundaries and Bayesian Mechanics

## Introduction

The concept of a Markov blanket is arguably the most foundational construct in the Free Energy Principle. It defines what it means to be a "thing" -- a system distinct from its environment. Without a Markov blanket, there is no boundary between organism and environment, no distinction between internal and external, and no basis for the FEP. Every application of the FEP begins with identifying the Markov blanket of the system in question.

## Definition and Origin

### Pearl's Markov Blanket (1988)

The concept was introduced by Judea Pearl in the context of Bayesian networks. Given a node `X` in a directed acyclic graph (DAG), the **Markov blanket** of `X` is the minimal set of nodes that renders `X` conditionally independent of all other nodes in the network.

For a DAG, the Markov blanket of `X` consists of:
1. **Parents** of `X`: nodes with directed edges into `X`
2. **Children** of `X`: nodes with directed edges from `X`
3. **Co-parents** of `X`'s children: other parents of `X`'s children

Formally:
```
X _||_ (V \ {X, MB(X)}) | MB(X)
```

Where `V` is the set of all nodes, `MB(X)` is the Markov blanket, and `_||_` denotes conditional independence.

**Interpretation**: If you know the state of every node in the Markov blanket, knowing additional nodes provides no further information about `X`. The blanket "screens off" `X` from the rest of the network.

### Friston's Markov Blanket (2013, 2019)

Friston adapted the Markov blanket concept for continuous dynamical systems. In this setting, the blanket is not defined on a static graph but on the dynamics of a coupled stochastic system.

Consider a system of coupled stochastic differential equations:
```
dx/dt = f(x) + w,  w ~ N(0, 2*Gamma)
```

Where `x = (psi, s, a, mu)` comprises external (`psi`), sensory (`s`), active (`a`), and internal (`mu`) states.

The Markov blanket is defined by the conditional independence structure of the system's steady-state density:

```
p(mu | psi, s, a) = p(mu | s, a)    (internal independent of external given blanket)
p(psi | mu, s, a) = p(psi | s, a)   (external independent of internal given blanket)
```

The blanket states `b = (s, a)` mediate ALL statistical dependencies between internal and external states.

## The Particular Partition

### Definition

The **particular partition** divides the state space of a system into four sets:

```
x = (psi, s, a, mu)
```

| Partition | Symbol | Description | Analogy |
|-----------|--------|-------------|---------|
| External states | psi | Environment; hidden from the organism | The world |
| Sensory states | s | Influenced by external states; observed by internal | Sensory receptors |
| Active states | a | Influenced by internal states; act on external | Motor effectors |
| Internal states | mu | The organism proper; screened from external by blanket | Brain/body |

The **blanket states** are `b = (s, a)`, and a **particular** is the combination of blanket and internal states: `(b, mu) = (s, a, mu)`.

### Conditional Independence Structure

The particular partition entails specific conditional independence properties encoded in the flow of the system. The dynamics respect:

```
f_mu(x) = f_mu(s, a, mu)     (internal flow depends only on blanket and internal states)
f_psi(x) = f_psi(s, a, psi)  (external flow depends only on blanket and external states)
f_s(x) = f_s(psi, s, a, mu)  (sensory flow can depend on everything)
f_a(x) = f_a(s, a, mu)       (active flow depends on blanket and internal states)
```

**Critical constraints**:
- Internal dynamics `f_mu` do NOT depend on external states `psi`
- Active dynamics `f_a` do NOT depend on external states `psi`
- External dynamics `f_psi` do NOT depend on internal states `mu`

This means:
- External states influence internal states ONLY through sensory states
- Internal states influence external states ONLY through active states
- The blanket mediates all exchange

### The Flow Diagram

```
              External (psi)
               |        ^
               v        |
           Sensory (s)  Active (a)
               |    \  / |
               v     \/  v
               |     /\  |
               v    /  \ v
           Internal (mu)
```

Arrows indicate causal influence. Note the asymmetry:
- Sensory states are caused by external states (and blanket states)
- Active states are caused by internal states (and blanket states)
- This creates a directed flow: external -> sensory -> internal -> active -> external

## Formal Derivation from Dynamics

### Setup: Langevin Dynamics

Consider a system governed by Langevin dynamics:

```
dx = f(x)dt + sigma * dW
```

Where `f(x)` is the flow (drift), `sigma` is the diffusion coefficient, and `dW` is a Wiener process.

At steady state, the system has a non-equilibrium steady-state (NESS) density `p(x)` satisfying the Fokker-Planck equation:

```
0 = -nabla . [f(x) * p(x)] + Gamma * nabla^2 p(x)
```

Where `Gamma = sigma * sigma^T / 2` is the diffusion tensor.

### The Helmholtz Decomposition

The flow `f(x)` can be decomposed via the Helmholtz decomposition:

```
f(x) = (Q - Gamma) * nabla ln p(x)
```

Where:
- `Q` is an antisymmetric matrix (solenoidal flow): `Q = -Q^T`
- `Gamma` is the diffusion tensor (dissipative flow): `Gamma = Gamma^T`

**Dissipative component**: `-(Gamma) * nabla ln p(x)` -- gradient flow that moves toward high-probability regions (like water flowing downhill). This component minimizes free energy.

**Solenoidal component**: `Q * nabla ln p(x)` -- divergence-free flow that circulates on iso-probability surfaces. This component does NOT change the probability density.

### Blanket Conditions on the Flow

The Markov blanket exists when certain entries of the flow coupling matrices are zero. Specifically, if we partition the Jacobian of the flow:

```
J = partial f / partial x = [[J_psi,psi  J_psi,s  J_psi,a  J_psi,mu],
                              [J_s,psi    J_s,s    J_s,a    J_s,mu  ],
                              [J_a,psi    J_a,s    J_a,a    J_a,mu  ],
                              [J_mu,psi   J_mu,s   J_mu,a   J_mu,mu ]]
```

The Markov blanket condition requires:
```
J_mu,psi = 0     (internal flow does not depend on external states)
J_psi,mu = 0     (external flow does not depend on internal states)
J_a,psi = 0      (active flow does not depend on external states)
```

These sparse coupling conditions create the conditional independence structure that defines the blanket.

## Blanket States and Free Energy Minimization

### The Fundamental Insight

Friston's key insight (2013, 2019) is that if a system has a Markov blanket and exists at a non-equilibrium steady state (NESS), then internal states can be described as performing approximate Bayesian inference about external states.

**The argument**:

1. At NESS, the flow of internal states is:
   ```
   f_mu = (Q_mu - Gamma_mu) * nabla_mu ln p(mu | b)
   ```
   Where `b = (s, a)` are blanket states.

2. Since `p(mu | b)` can be decomposed:
   ```
   ln p(mu | b) = ln p(psi | b) + ln p(mu | psi, b) - [terms not depending on mu]
   ```

   But by the Markov blanket condition: `p(mu | psi, b) = p(mu | b)`, so this simplifies.

3. The steady-state density of internal states `p(mu | b)` plays the role of a recognition density `q(psi) = q(psi | mu, b)` -- a mapping from internal states to beliefs about external states.

4. The gradient flow `f_mu proportional to nabla_mu ln p(mu | b)` can be rewritten as:
   ```
   f_mu proportional to -nabla_mu F
   ```
   Where `F` is a variational free energy functional.

**Conclusion**: Internal states perform gradient descent on variational free energy as a natural consequence of their dynamics at steady state. They do not need to "compute" free energy -- their dynamics ARE free energy minimization.

### The Recognition Density

The mapping from internal states to beliefs about external states is called the **recognition density**:

```
q(psi) = q(psi | mu)
```

Under the Laplace approximation (Gaussian assumption), this becomes:

```
q(psi) = N(mu_psi(mu), Sigma_psi(mu))
```

Where `mu_psi(mu)` maps internal states to the expected external state and `Sigma_psi(mu)` maps to the uncertainty.

The internal states are thus a **parameterization** of beliefs about external states. The brain's neural activity patterns are not arbitrary -- they parameterize probability distributions over hidden causes of sensory input.

## Pearl Blankets vs. FEP Blankets

There is an important distinction between Markov blankets as used by Pearl and as used in the FEP literature:

| Aspect | Pearl Blanket | FEP Blanket |
|--------|--------------|-------------|
| Context | Bayesian networks (DAGs) | Continuous dynamical systems |
| Definition | Graph-theoretic (parents, children, co-parents) | Dynamic (conditional independence in flow) |
| Stationarity | Static structure | Emerges at steady state |
| Directionality | Defined by DAG edges | Defined by sensory/active asymmetry |
| Scale | Single node | Nested hierarchies possible |
| Uniqueness | Unique for a given DAG | Depends on partition of dynamics |

### Controversies

The interpretation of Markov blankets in the FEP has generated debate:

1. **Ontological vs. epistemic status**: Are Markov blankets real physical boundaries or merely useful descriptions? Kirchhoff et al. (2018) argue they are ontological; Bruineberg et al. (2022) argue they are epistemic constructs.

2. **Existence conditions**: Not all dynamical systems have well-defined Markov blankets. The conditions for blanket existence are non-trivial and may not hold for arbitrary systems.

3. **Blanket persistence**: The FEP requires blankets to persist over time, but blankets can form and dissolve (e.g., a soap bubble). How do we handle transient blankets?

4. **Sparse coupling assumption**: The zero entries in the Jacobian (e.g., `J_mu,psi = 0`) are a strong assumption. In practice, these entries may be approximately but not exactly zero.

## Nested Markov Blankets

### Multi-Scale Organization

One of the most powerful features of the Markov blanket framework is that blankets can be **nested**. A cell has a Markov blanket (its membrane). An organ composed of cells has its own Markov blanket. An organism composed of organs has its own blanket. And so on.

```
Social group
  └── Organism (blanket: skin, sensory organs, muscles)
       └── Organ (blanket: organ boundary, vasculature)
            └── Cell (blanket: cell membrane, receptors, effectors)
                 └── Organelle (blanket: organelle membrane)
```

At each level:
- Internal states at level n become external states at level n-1
- Blanket states mediate between levels
- Free energy minimization occurs at each level simultaneously

### Formal Nesting

If states `x` can be partitioned into `(x_1, x_2, ..., x_N)` where each `x_i` is itself a particular with its own blanket, then:

```
x_i = (psi_i, s_i, a_i, mu_i)
```

The higher-level states emerge as functions of the lower-level states:

```
X = phi(x_1, x_2, ..., x_N)
```

Where `phi` is a coarse-graining map. The higher-level system has its own Markov blanket in terms of these coarse-grained variables.

**This explains multi-scale self-organization**: cells organize into tissues, tissues into organs, organs into organisms, organisms into societies -- each level has its own blanket and minimizes free energy at its own scale.

## Bayesian Mechanics

### Definition

**Bayesian mechanics** (Ramstead et al., 2023) is the formal study of systems that possess Markov blankets and can be described as performing inference. It is the physics of systems that "look as if" they are doing Bayesian inference by virtue of possessing a particular partition.

### The Path of Particular Physics

The key equations of Bayesian mechanics describe how internal states evolve:

```
dmu/dt = f_mu(b, mu) = (Q_mu - Gamma_mu) * nabla_mu F_mu
```

Where `F_mu` is the free energy as a function of internal states. This says:

1. Internal states flow in the direction that reduces free energy (dissipative part)
2. Internal states also circulate on free energy iso-surfaces (solenoidal part)
3. The steady state of internal dynamics is the free energy minimum

### Synchronization Map

The **synchronization map** `sigma: mu -> psi*` maps internal states to the external state they are "synchronized with" -- the external state that would be most probable given the current internal and blanket states:

```
sigma(mu) = argmax_psi p(psi | b, mu) = argmax_psi p(psi | b)
```

Under the Laplace approximation, this is simply the mode of the conditional:

```
sigma(mu) = E[psi | b]
```

When internal dynamics have converged, `sigma(mu)` tracks the actual external state `psi` -- the organism's beliefs are synchronized with reality.

## Examples of Markov Blankets

### Biological Examples

| System | Internal States | Sensory States | Active States | External States |
|--------|----------------|----------------|---------------|-----------------|
| **Cell** | Cytoplasm, DNA | Membrane receptors | Ion channels, secretion | Extracellular medium |
| **Neuron** | Intracellular potential, proteins | Dendrites, synapses | Axon terminal, neurotransmitter release | Pre-synaptic neurons, glial cells |
| **Brain** | Neural populations | Sensory neurons | Motor neurons | Body and environment |
| **Organism** | Brain, organs | Sensory organs (eyes, ears, skin) | Muscles, glands | Environment |
| **Colony** | Individual organisms | Scouts, sentinels | Workers, soldiers | Ecosystem |

### Physical Examples

- **Benard cell**: Internal states = convection pattern; blanket = cell boundary; external = thermal gradient
- **Flame**: Internal states = combustion chemistry; blanket = reaction zone; external = fuel and oxygen
- **Hurricane**: Internal states = wind circulation; blanket = eye wall; external = ocean and atmosphere

### Artificial Examples

- **Thermostat**: Internal = temperature estimate; sensory = thermometer; active = heater switch; external = room temperature
- **Robot**: Internal = state estimates; sensory = cameras, LIDAR; active = motors; external = physical environment
- **Neural network**: Internal = hidden layers; sensory = input layer; active = output layer; external = training data

## Mathematical Properties

### Conditional Independence Lemma

Given the particular partition `(psi, s, a, mu)`:

```
mu _||_ psi | b  <=>  p(mu, psi | b) = p(mu | b) * p(psi | b)
```

This means the joint distribution of internal and external states, conditioned on blanket states, factorizes. The mutual information between internal and external states, conditioned on blanket states, is zero:

```
I(mu; psi | b) = 0
```

### Free Energy Gradient Identity

At NESS, the flow of internal states satisfies:

```
f_mu = (Q_mu,mu - Gamma_mu,mu) * nabla_mu ln p(mu | b)
     = -(Q_mu,mu - Gamma_mu,mu) * nabla_mu F[q(psi | mu), o(b)]
```

This identity connects the dynamics of internal states to gradient descent on free energy, establishing the FEP as a consequence of steady-state dynamics with a Markov blanket.

### Blanket State Dynamics

The dynamics of blanket states are more complex because they are influenced by both internal and external states:

```
f_s = f_s(psi, b)      (sensory flow depends on external and blanket)
f_a = f_a(mu, b)       (active flow depends on internal and blanket)
```

Sensory states carry information from external to internal (perception pathway).
Active states carry influence from internal to external (action pathway).

## Open Questions and Active Research

1. **Blanket identification**: How to identify Markov blankets in real empirical data? Methods include transfer entropy analysis, Granger causality, and information-theoretic measures.

2. **Approximate blankets**: Real systems may have "leaky" blankets where conditional independence is approximate. How does the FEP degrade under approximate blankets?

3. **Blanket dynamics**: Blankets themselves can change (grow, shrink, merge, split). A theory of blanket dynamics would capture morphogenesis, cell division, death, and social group formation.

4. **Quantum blankets**: Can the Markov blanket framework be extended to quantum systems where the notion of "state" is fundamentally different?

5. **Computational blankets**: In artificial neural networks, what constitutes the Markov blanket? How does this relate to modularity, information bottlenecks, and disentangled representations?

## Key References

1. Pearl, J. (1988). *Probabilistic Reasoning in Intelligent Systems*. Morgan Kaufmann.
2. Friston, K. (2013). Life as we know it. *Journal of the Royal Society Interface*, 10(86), 20130475.
3. Friston, K. (2019). A free energy principle for a particular physics. *arXiv preprint* arXiv:1906.10184.
4. Kirchhoff, M., Parr, T., Palacios, E., Friston, K., & Kiverstein, J. (2018). The Markov blankets of life: autonomy, active inference and the free energy principle. *Journal of the Royal Society Interface*, 15(138), 20170792.
5. Bruineberg, J., Dolega, K., Dewhurst, J., & Baltieri, M. (2022). The emperor's new Markov blankets. *Behavioral and Brain Sciences*, 45, e183.
6. Ramstead, M. J. D., Sakthivadivel, D. A. R., Heins, C., Koudahl, M., Millidge, B., Da Costa, L., Klein, B., & Friston, K. J. (2023). On Bayesian mechanics: a physics of and by beliefs. *Interface Focus*, 13(3), 20220029.
7. Parr, T., Da Costa, L., & Friston, K. (2020). Markov blankets, information geometry and stochastic thermodynamics. *Philosophical Transactions of the Royal Society A*, 378(2164), 20190159.

## See also

- [[knowledge_base/mathematics/markov_blankets|Markov Blankets]]
