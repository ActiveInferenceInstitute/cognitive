---
title: Markov Blankets
type: concept
status: active
created: 2025-02-06
updated: 2025-02-06
tags:
  - markov-blanket
  - conditional-independence
  - bayesian-mechanics
  - free-energy-principle
  - particular-partition
semantic_relations:
  - type: foundation
    links:
      - [[mathematics/probability_theory|Probability Theory]]
      - [[mathematics/bayesian_networks|Bayesian Networks]]
      - [[mathematics/conditional_independence|Conditional Independence]]
  - type: relates
    links:
      - [[cognitive/active_inference|Active Inference]]
      - [[cognitive/free_energy_principle|Free Energy Principle]]
      - [[mathematics/non_equilibrium_steady_state|Non-Equilibrium Steady State]]
  - type: implements
    links:
      - [[mathematics/probabilistic_graphical_models|Probabilistic Graphical Models]]
      - [[mathematics/variational_free_energy|Variational Free Energy]]
---

# Markov Blankets

## Overview

A Markov blanket is a statistical concept that defines the boundary of a system in terms of conditional independence: the Markov blanket of a set of variables renders those variables conditionally independent of all other variables in the system. In the free energy principle (FEP), Markov blankets play a foundational role -- they define the boundary between an organism (or any self-organizing system) and its environment, enabling the formal definition of what it means for a system to "be" something distinct from its surroundings. The blanket separates internal states (which encode beliefs about external states) from external states (the environment), with the blanket itself comprising sensory states (through which the environment influences the system) and active states (through which the system influences the environment).

## Formal Definition

### Pearl's Original Definition

The concept originates with Judea Pearl (1988) in the context of Bayesian networks. For a node `X` in a directed acyclic graph (DAG), the Markov blanket `MB(X)` consists of:

1. **Parents** of X: `pa(X)`
2. **Children** of X: `ch(X)`
3. **Other parents of children** of X (co-parents): `pa(ch(X)) \ {X}`

Formally:

```math
MB(X) = pa(X) \cup ch(X) \cup pa(ch(X)) \setminus \{X\}
```

The defining property is conditional independence:

```math
X \perp\!\!\!\perp (U \setminus MB(X) \setminus \{X\}) \mid MB(X)
```

where `U` is the set of all variables in the graph. Given the Markov blanket, X is independent of every other variable in the system.

### Undirected Formulation

In undirected graphical models (Markov random fields), the Markov blanket of a node X is simply its set of neighbors in the graph:

```math
MB(X) = \{Y : Y \sim X\}
```

where `Y ~ X` denotes that Y and X are connected by an edge.

### General Definition

In the most general setting, the Markov blanket of a set of variables `I` (internal states) with respect to a joint distribution `P(I, B, E)` over internal states I, blanket states B, and external states E is a set B such that:

```math
P(I | B, E) = P(I | B)
```

equivalently:

```math
I \perp\!\!\!\perp E \mid B
```

Internal states are conditionally independent of external states given the blanket states.

## Conditional Independence Properties

### Screening Off

The Markov blanket "screens off" internal from external states. All information that external states carry about internal states is mediated through the blanket:

```math
I(I; E | B) = 0
```

where `I(.; .)` denotes mutual information. This means:
- Internal states need only track blanket states to maintain optimal beliefs about external states
- External states can only influence internal states through the blanket
- Internal states can only influence external states through the blanket

### Factorization

The conditional independence implied by the Markov blanket enables a factorization of the joint distribution:

```math
P(I, B, E) = P(I | B) P(E | B) P(B)
```

This factorization is key to the FEP: it means that the internal dynamics can be understood as a function of blanket states alone, without direct reference to external states.

### Information-Theoretic Characterization

The Markov blanket can be characterized as the minimal set of variables that preserves all information about internal states:

```math
MB(I) = \arg\min_{B \subseteq U \setminus I} |B| \quad \text{subject to } I(I; U \setminus I \setminus B | B) = 0
```

This minimality condition ensures the blanket contains no redundant variables.

## The Particular Partition

### Friston's Extension

Karl Friston (2013, 2019) introduced the "particular partition" -- a specific decomposition of a system's states into four sets based on their causal relationships:

```math
\mathcal{X} = \{\mu, s, a, \eta\}
```

where:
- **mu (internal states)**: States that constitute the system's "beliefs" or sufficient statistics
- **s (sensory states)**: Blanket states influenced by external states but not by internal states
- **a (active states)**: Blanket states influenced by internal states but not by external states
- **eta (external states)**: States outside the blanket, constituting the "environment"

The blanket states are the union of sensory and active states:

```math
b = s \cup a
```

### Causal Structure

The particular partition implies a specific causal structure:

```
External (eta) --> Sensory (s) --> Internal (mu) --> Active (a) --> External (eta)
```

This creates a circular causal flow:
1. External states cause changes in sensory states
2. Sensory states influence internal states (perception)
3. Internal states influence active states (action)
4. Active states influence external states (environmental modification)

### Formal Dynamics

Under the particular partition, the dynamics of each partition element can be written as:

```math
\dot{\mu} = f_\mu(\mu, s, a) + \omega_\mu
```
```math
\dot{s} = f_s(s, a, \eta) + \omega_s
```
```math
\dot{a} = f_a(\mu, s, a) + \omega_a
```
```math
\dot{\eta} = f_\eta(s, a, \eta) + \omega_\eta
```

where `f` denotes the deterministic flow and `omega` denotes random fluctuations. Crucially:
- Internal state dynamics `f_mu` do not depend on external states `eta`
- External state dynamics `f_eta` do not depend on internal states `mu`
- All coupling between internal and external goes through the blanket `(s, a)`

### Autonomous States

The autonomous states are defined as the union of internal and active states:

```math
\alpha = \mu \cup a
```

These are the states whose dynamics depend on (and only on) internal and blanket states. Autonomous states are "owned" by the system in the sense that their dynamics are determined by the system's own states and its sensory interface with the world.

## Pearl Blankets vs. Friston Blankets

### Key Differences

The distinction between Pearl blankets and Friston blankets has been a subject of technical debate:

| Feature | Pearl Blanket | Friston Blanket |
|---------|--------------|-----------------|
| **Context** | Static Bayesian networks | Dynamical systems at NESS |
| **Definition** | Graph-theoretic (parents, children, co-parents) | Dynamical (causal flow in SDEs) |
| **Interpretation** | Statistical screening | Physical boundary of self-organization |
| **Directionality** | Blanket has no internal structure | Blanket partitions into sensory and active |
| **Time** | Atemporal (single snapshot) | Temporal (defined over dynamics) |

### Pearl Blankets

Pearl blankets are defined graph-theoretically. Given a Bayesian network, the Markov blanket of any node can be read off from the graph structure. These blankets:
- Are well-defined for any probabilistic graphical model
- Provide conditional independence guarantees
- Are minimal by definition
- Do not require any physical or causal interpretation

### Friston Blankets

Friston blankets (sometimes called "Friston-Pearl blankets" or "active blankets") are defined dynamically. They arise from the particular partition of a stochastic dynamical system at a non-equilibrium steady state (NESS). These blankets:
- Require a dynamical system with a steady-state density
- Are defined through the sparsity structure of the system's Jacobian or coupling
- Have internal structure (sensory/active partition) with causal directionality
- Support the interpretation of internal states as performing inference about external states

### The Sparse Coupling Requirement

Friston blankets require sparse coupling -- not all states interact with all other states. The blanket exists because the system's dynamics exhibit a specific pattern of conditional independence that can be identified from the zero entries in the system's coupling matrix (Jacobian):

```math
J = \begin{pmatrix} J_{\mu\mu} & J_{\mu s} & J_{\mu a} & 0 \\ 0 & J_{ss} & J_{sa} & J_{s\eta} \\ J_{a\mu} & J_{as} & J_{aa} & 0 \\ 0 & J_{\eta s} & J_{\eta a} & J_{\eta\eta} \end{pmatrix}
```

The zeros in positions (mu, eta) and (eta, mu) encode the conditional independence that defines the blanket.

## Bayesian Mechanics

### The Inference Interpretation

Bayesian mechanics (Friston, 2019; Sakthivadivel, 2022) is the study of systems that can be described "as if" their internal states perform inference about external states. The Markov blanket is the key structure that enables this interpretation:

Given the particular partition and NESS density `p(mu, s, a, eta)`, internal states parametrize a probability distribution over external states:

```math
q_\mu(\eta) \triangleq p(\eta | b) \quad \text{where } b = (s, a) \text{ at the current blanket state}
```

The dynamics of internal states can then be interpreted as updating this belief:

```math
\dot{\mu} = -\nabla_\mu F(\mu, s, a)
```

where `F` is the variational free energy. This is the formal basis for the claim that any system with a Markov blanket can be described as performing (approximate) Bayesian inference.

### Path-Based Blankets

Recent work (Friston et al., 2023) has extended the Markov blanket concept to path space -- considering not just the instantaneous configuration of states but entire trajectories. A path-based Markov blanket separates internal and external trajectories, enabling a more general formulation of Bayesian mechanics that does not require steady-state assumptions.

### The Free Energy Lemma

The free energy lemma (Friston, 2019) states that for any system with a Markov blanket at NESS, the internal states minimize a functional of blanket states that is bounded below by negative log-evidence:

```math
F(\mu, b) \geq -\ln p(b)
```

This means internal dynamics are guaranteed to perform approximate inference in the sense that they minimize an upper bound on surprisal.

## Sparse Coupling

### Physical Basis

Markov blankets arise naturally in physical systems due to spatial locality -- physical interactions typically involve nearest neighbors. The range of physical forces ensures that sufficiently distant components are conditionally independent given intermediate components.

### Examples of Sparse Coupling

1. **Spatial systems**: In a lattice model, each site interacts with its neighbors. The blanket of an interior region is the set of sites on its boundary.

2. **Chemical systems**: In a well-mixed reactor, all species interact. But in spatially extended systems, diffusion-limited reactions create local coupling that admits blankets.

3. **Network systems**: In sparse networks (small-world, scale-free), the low average degree ensures that most nodes have compact Markov blankets.

### The Blanket as Physical Membrane

In biological systems, the Markov blanket often corresponds to a physical membrane or boundary:
- Cell membrane separates intracellular from extracellular states
- Skin separates organism from environment
- Blood-brain barrier separates neural from metabolic states

However, the blanket need not be a physical membrane -- it is a statistical concept defined by conditional independence. In social systems, for example, the blanket might be constituted by communicative actions rather than physical barriers.

## Blankets in Biological Systems

### Cellular Level

The paradigmatic biological Markov blanket is the cell membrane:
- **Internal states**: Intracellular biochemistry (gene expression, protein concentrations, metabolic states)
- **Sensory states**: Receptor proteins, ion channels (sense extracellular signals)
- **Active states**: Secretory pathways, membrane pumps (modify extracellular environment)
- **External states**: Extracellular environment (other cells, tissue fluid, signaling molecules)

### Neural Level

At the neural level, Markov blankets can be identified at multiple scales:
- **Single neuron**: Dendritic inputs (sensory), axonal outputs (active), soma (internal)
- **Cortical column**: Input layer IV (sensory), output layers II/III and V/VI (active), layers II-VI (internal)
- **Brain region**: Afferent connections (sensory), efferent connections (active), local circuitry (internal)

### Organism Level

For a whole organism:
- **Internal states**: Central nervous system, internal organs
- **Sensory states**: Sensory receptors, interoceptors
- **Active states**: Motor outputs, secretory outputs, behavioral actions
- **External states**: Physical and social environment

### Social Level

At the social level, Markov blankets may describe:
- **Individuals within groups**: Communication and observation (sensory), social actions (active)
- **Institutions**: Information intake (sensory), policy implementation (active), internal deliberation (internal)
- **Nations**: Diplomacy and intelligence (sensory), foreign policy and trade (active)

## Blankets at Multiple Scales

### Hierarchical Nesting

A key feature of Markov blankets in biological systems is their hierarchical nesting: blankets within blankets within blankets. Each level of biological organization possesses its own blanket:

```
Organelle blanket ⊂ Cell blanket ⊂ Tissue blanket ⊂ Organ blanket ⊂ Organism blanket ⊂ Social group blanket
```

At each level, internal states at one level become the substrate for blanket structure at the next level up.

### Multi-Scale Free Energy Minimization

Hierarchically nested blankets imply multi-scale free energy minimization. Each level minimizes its own free energy, subject to constraints imposed by adjacent levels:

```math
F_{total} = \sum_{l=1}^{L} F_l(\mu_l, b_l)
```

Lower levels provide fast, local inference while higher levels provide slow, global context. This hierarchical structure is a natural consequence of the sparse coupling that arises from physical locality and the multi-scale organization of biological matter.

### Emergence Through Blanket Nesting

The emergence of higher-level entities (cells from molecules, organisms from cells, societies from organisms) can be formalized through Markov blanket nesting. A higher-level entity "exists" in the blanket sense when a set of lower-level entities forms a collective blanket that screens off collective internal states from collective external states.

## Mathematical Formalization

### Measure-Theoretic Definition

In measure-theoretic terms, given a probability space `(Omega, F, P)` and sigma-algebras `F_I, F_B, F_E ⊂ F` corresponding to internal, blanket, and external states respectively, the blanket condition is:

```math
\mathbb{E}[f(I) g(E) | B] = \mathbb{E}[f(I) | B] \cdot \mathbb{E}[g(E) | B]
```

for all measurable functions f and g. This is the conditional independence `I ⊥ E | B` in measure-theoretic language.

### Information Geometry

The Markov blanket defines a geometric structure on the space of internal states. Each configuration of internal states corresponds to a point in the space of probability distributions over external states (via the conditional distribution `p(eta | b)`). The Fisher information metric on this space provides a natural Riemannian geometry:

```math
g_{ij}(\mu) = \mathbb{E}_{q_\mu(\eta)}\left[\frac{\partial \ln q_\mu(\eta)}{\partial \mu_i} \frac{\partial \ln q_\mu(\eta)}{\partial \mu_j}\right]
```

This geometric structure connects Markov blankets to information geometry and provides a natural metric for measuring the "distance" between beliefs.

### Existence Conditions

Not all dynamical systems possess Markov blankets. Sufficient conditions include:

1. **Sparse coupling**: The system's interaction structure must have sufficient sparsity
2. **Steady state**: The system must possess a non-equilibrium steady state (or at least a slowly evolving quasi-steady state)
3. **Ergodicity**: The system must explore its state space sufficiently to define a steady-state density
4. **Smoothness**: The dynamics must be sufficiently regular (typically Lipschitz continuous)

## Examples

### Example 1: Simple Two-Oscillator System

Consider two coupled oscillators with states `x_1` and `x_2`:

```math
\dot{x}_1 = -\alpha x_1 + \kappa s + \omega_1
```
```math
\dot{s} = -\beta s + x_1 + x_2 + \omega_s
```
```math
\dot{x}_2 = -\alpha x_2 + \kappa s + \omega_2
```

Here `s` is a shared state (blanket), `x_1` and `x_2` are conditionally independent given `s`. Each oscillator's blanket relative to the other is `{s}`.

### Example 2: Neural Network

In a feedforward neural network with layers `L_1, L_2, L_3`:

```math
L_1 \rightarrow L_2 \rightarrow L_3
```

The Markov blanket of `L_1` with respect to `L_3` is `{L_2}`. Layer 2 screens off layer 1 from layer 3.

### Example 3: Cell Membrane

For a cell with intracellular concentration `c_in`, membrane transport rate `m`, and extracellular concentration `c_out`:

```math
\dot{c}_{in} = -k_1 c_{in} + k_2 m + \omega_{in}
```
```math
\dot{m} = f(c_{in}, c_{out}) + \omega_m
```
```math
\dot{c}_{out} = -k_3 c_{out} + k_4 m + \omega_{out}
```

The membrane state `m` forms a Markov blanket separating internal `c_in` from external `c_out`.

## Criticisms and Debates

### The "Every-Thing" Problem

A common criticism is that Markov blankets can be drawn around arbitrary subsets of variables in any system with sparse coupling, leading to the concern that the FEP trivially applies to "every thing" without providing specific predictions. Responses include:
- The blanket must persist over time (temporal stability)
- The internal states must be functionally organized to minimize free energy (not just any partition qualifies)
- The interesting claim is about the dynamics, not just the existence of a boundary

### Static vs. Dynamic Blankets

Some authors distinguish between static (instantaneous) and dynamic (persistent) Markov blankets. The FEP requires dynamic blankets -- boundaries that persist over time as the system evolves. This is a stronger condition than merely finding a partition at a single time point.

### Blankets and Identity

The question of whether Markov blankets define the "identity" of a system has philosophical implications. If a system's boundary is statistical rather than physical, what determines where one system ends and another begins? This connects to questions about personal identity, consciousness, and the nature of biological individuality.

## Key References

- Pearl, J. (1988). Probabilistic Reasoning in Intelligent Systems. Morgan Kaufmann.
- Friston, K. (2013). Life as we know it. Journal of the Royal Society Interface, 10(86), 20130475.
- Friston, K. (2019). A free energy principle for a particular physics. arXiv:1906.10184.
- Kirchhoff, M., et al. (2018). The Markov blankets of life. Journal of the Royal Society Interface, 15(138).
- Sakthivadivel, D. (2022). Towards a geometry and analysis for Bayesian mechanics. arXiv:2204.11900.
- Bruineberg, J., et al. (2022). The emperor's new Markov blankets. Behavioral and Brain Sciences, 45, e183.

## Cross-References

- [[mathematics/conditional_independence|Conditional Independence]] - Core mathematical property
- [[mathematics/bayesian_networks|Bayesian Networks]] - Graph-theoretic context
- [[cognitive/active_inference|Active Inference]] - Framework that depends on blanket structure
- [[cognitive/free_energy_principle|Free Energy Principle]] - Theoretical framework
- [[mathematics/non_equilibrium_steady_state|Non-Equilibrium Steady State]] - Required for dynamical blankets
- [[mathematics/fokker_planck|Fokker-Planck Equation]] - Density dynamics at NESS
- [[cognitive/embodied_cognition|Embodied Cognition]] - Blankets as body boundaries
- [[biology/morphogenesis|Morphogenesis]] - Blankets in developmental biology
