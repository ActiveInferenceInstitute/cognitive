---
title: Circular Causality
type: concept
status: active
created: 2025-02-06
updated: 2025-02-06
tags:
  - circular-causality
  - top-down-causation
  - bottom-up-causation
  - synergetics
  - strange-loops
  - emergence
  - predictive-processing
semantic_relations:
  - type: foundation
    links:
      - [[systems/emergence|Emergence]]
      - [[systems/dynamical_systems|Dynamical Systems]]
      - [[systems/synergetics|Synergetics]]
  - type: relates
    links:
      - [[cognitive/free_energy_principle|Free Energy Principle]]
      - [[cognitive/predictive_coding|Predictive Coding]]
      - [[cognitive/hierarchical_inference|Hierarchical Inference]]
      - [[mathematics/renormalization_group|Renormalization Group]]
  - type: extends
    links:
      - [[philosophy/enactivism|Enactivism]]
      - [[mathematics/markov_blankets|Markov Blankets]]
---

# Circular Causality

## Overview

Circular causality refers to the reciprocal causal relationship between levels of a hierarchical system, where microscopic components give rise to macroscopic patterns (bottom-up causation) and macroscopic patterns constrain the behavior of microscopic components (top-down causation). This circular relationship -- where the whole both emerges from and constrains its parts -- is fundamental to understanding complex adaptive systems, from neural circuits to ecosystems to societies. Under the free energy principle (FEP), circular causality is formalized through hierarchical generative models and the bidirectional flow of information (predictions and prediction errors) between levels of the cortical hierarchy.

## Top-Down and Bottom-Up Causation

### Bottom-Up Causation

Bottom-up causation (upward causation) is the process by which microscopic dynamics produce macroscopic patterns:

- **Molecular interactions** produce cellular behavior
- **Neural firing patterns** produce cognitive states
- **Individual behaviors** produce social phenomena
- **Genetic expression** produces organismic traits

Formally, bottom-up causation maps microscopic states to macroscopic observables:

```math
\text{Macro state} = \Phi[\text{micro states}]
```

where `Phi` is a coarse-graining function that projects the high-dimensional microscopic state into a lower-dimensional macroscopic description.

### Top-Down Causation

Top-down causation (downward causation) is the process by which macroscopic patterns constrain microscopic dynamics:

- **Cellular context** constrains gene expression
- **Cognitive states** constrain neural firing
- **Social norms** constrain individual behavior
- **Environmental conditions** constrain evolutionary trajectories

Top-down causation does not mean that macro states directly push micro states around (which would violate causal closure at the physical level). Instead, it means that the macroscopic pattern creates boundary conditions, constraints, or contexts that select among possible microscopic dynamics:

```math
P(\text{micro state} | \text{macro state}) \neq P(\text{micro state})
```

The macro state changes the probability distribution over micro states without directly intervening in the micro dynamics.

### The Circularity

The causal circularity arises because:
1. Micro states produce the macro state (bottom-up)
2. The macro state constrains micro states (top-down)
3. The constrained micro states produce an updated macro state (bottom-up again)
4. The updated macro state further constrains micro states (top-down again)

This creates a self-sustaining loop:

```
Micro states -> (emergence) -> Macro state -> (constraint) -> Micro states -> ...
```

The loop has no "first cause" -- neither the micro nor the macro level is primary. Both are aspects of a single self-organizing process.

## Reciprocal Influence

### Co-Determination

In circular causality, micro and macro levels co-determine each other. Neither level can be understood in isolation:

- The macro pattern only exists because of micro-level dynamics
- The micro-level dynamics only take their observed form because of macro-level constraints
- Changing either level changes the other
- The system's behavior is a property of the whole, not reducible to either level

### Examples of Reciprocal Influence

1. **Convection cells**: Individual molecule motions (micro) create convection patterns (macro). The convection pattern constrains molecular motion (molecules move in organized streams, not randomly). The organized molecular motion maintains the convection pattern.

2. **Neural oscillations**: Individual neuron firing (micro) creates oscillatory patterns (macro). The oscillatory pattern constrains neural firing (neurons tend to fire in phase with the oscillation). The phase-locked firing maintains the oscillation.

3. **Market dynamics**: Individual trades (micro) create market trends (macro). Market trends constrain individual trading behavior (trend-following, fear of loss). Individual responses to trends further shape market dynamics.

4. **Language**: Individual speech acts (micro) create linguistic conventions (macro). Linguistic conventions constrain individual speech (grammar, vocabulary). Individual innovation within conventions drives language evolution.

## Reentrant Processing

### Definition

Reentrant processing (Edelman, 1989) is the continuous reciprocal signaling between brain regions (or hierarchical levels). Unlike simple feedback (where information flows in a loop), reentrant processing involves massive parallel reciprocal connections:

```
Region A <--> Region B <--> Region C
    |              |              |
    v              v              v
Region D <--> Region E <--> Region F
```

Every region both sends to and receives from multiple other regions simultaneously.

### Reentrant Processing and Circular Causality

Reentrant processing implements circular causality in the brain:
- **Bottom-up**: Sensory information flows from lower to higher cortical areas
- **Top-down**: Predictions flow from higher to lower areas
- **Lateral**: Context flows between areas at the same level

The continuous exchange of signals across all these pathways creates the self-sustaining dynamics that constitute perception, cognition, and consciousness.

### Reentrant Processing in Predictive Coding

In the predictive coding framework, reentrant processing takes a specific form:
- **Top-down connections carry predictions**: Higher areas send predictions of lower-area activity
- **Bottom-up connections carry prediction errors**: Lower areas send the discrepancy between prediction and observation
- **The cycle**: Predictions generate expectations, errors refine expectations, refined expectations generate new predictions

This prediction-error cycle is circular causality in its purest neural form.

## Strange Loops

### Hofstadter's Strange Loops

Douglas Hofstadter (1979, 2007) introduced the concept of "strange loops" -- self-referential structures where moving through levels of a hierarchy eventually returns to the starting point:

- Escher's drawing hands (each hand draws the other)
- Godel's incompleteness (a formal system that refers to itself)
- Consciousness (a brain that models itself)

### Strange Loops in the FEP

The FEP involves a strange loop at its core:
1. The organism models the world
2. The world includes the organism
3. The organism therefore models itself
4. This self-model influences the organism's actions
5. These actions change the world (including the organism)
6. The changed world changes the self-model
7. Return to step 1

This self-referential structure is not a defect but a defining feature of autonomous systems. The organism's identity is constituted by the loop itself, not by any fixed substrate.

### Self-Modeling and Consciousness

Strange loops in the FEP may be relevant to consciousness:
- The organism's generative model includes a model of itself
- This self-model generates predictions about the organism's own states (metacognition)
- The self-model's predictions can conflict with actual states (creating metacognitive prediction errors)
- The dynamics of self-modeling may be what gives rise to subjective experience

## Connection to Predictive Processing Hierarchy

### Hierarchical Predictive Coding

The cortical hierarchy implements circular causality through bidirectional message passing:

```
Level L (abstract): Generates predictions about Level L-1
    |  ^
    v  |
Level L-1: Sends prediction errors to Level L
    |  ^     Generates predictions about Level L-2
    v  |
Level L-2: Sends prediction errors to Level L-1
    ...
Level 0 (sensory): Receives observations from the world
```

At each level:
- **Bottom-up** (ascending): Prediction errors from the level below
- **Top-down** (descending): Predictions to the level below
- **Lateral** (within-level): Contextual modulation

### Dynamics of Hierarchical Inference

The dynamical equations for hierarchical predictive coding encode circular causality explicitly:

```math
\dot{\mu}_i = D\mu_i - \frac{\partial F}{\partial \mu_i} = D\mu_i + \pi_{i+1} \varepsilon_{i+1} - \pi_i \varepsilon_i
```

where:
- `mu_i` is the expectation at level i
- `epsilon_i` is the prediction error at level i (bottom-up signal)
- `pi_i` is the precision at level i
- `D` is a temporal derivative operator

The two terms on the right encode circular causality:
- `pi_{i+1} epsilon_{i+1}`: Bottom-up influence (prediction errors from below drive belief updating)
- `pi_i epsilon_i`: Top-down influence (predictions from above constrain the expected prediction error)

### Temporal Hierarchy

Circular causality also operates across timescales:
- **Fast dynamics** (milliseconds): Perceptual inference at lower levels
- **Medium dynamics** (seconds): Attention and contextual inference
- **Slow dynamics** (minutes-hours): Learning and model updating
- **Very slow dynamics** (days-years): Developmental and structural change

Each timescale influences the others: fast perceptual dynamics are shaped by slow learning, and slow learning is driven by the statistics of fast perceptual dynamics.

## Autopoiesis

### Circular Causality in Living Systems

Autopoiesis (Maturana & Varela, 1980) is the paradigmatic example of circular causality in biology:

1. The cell membrane (macro) is produced by intracellular processes (micro)
2. The membrane constrains intracellular processes (by creating a bounded volume, maintaining concentration gradients, selectively importing/exporting)
3. The constrained processes maintain the membrane (continuously replacing lipids, proteins)
4. The maintained membrane continues to constrain processes

This loop has no beginning and no end -- it is the loop itself that constitutes the living system.

### Autopoiesis and the FEP

Under the FEP, the autopoietic loop is formalized as:
- **Internal states** produce **active states** (internal dynamics drive boundary maintenance)
- **Active states** influence **external states** (boundary modifications change the environment)
- **External states** produce **sensory states** (environmental changes are sensed)
- **Sensory states** influence **internal states** (sensory input drives internal dynamics)

The Markov blanket (sensory + active states) mediates the circular causality between internal and external states.

## Emergence Through Circular Causality

### Strong vs. Weak Emergence

Circular causality is central to the debate about emergence:

**Weak emergence**: Macro properties are unexpected but in principle derivable from micro properties. Bottom-up causation is sufficient; top-down causation is merely a useful description.

**Strong emergence**: Macro properties are genuinely novel and not derivable from micro properties alone. Top-down causation is real and necessary for explanation.

### Circular Causality Transcends the Debate

Circular causality suggests that the strong/weak distinction may be a false dichotomy:
- The macro pattern is produced by micro dynamics (consistent with weak emergence)
- But the macro pattern constrains micro dynamics in ways that cannot be predicted from micro dynamics alone (consistent with strong emergence)
- The system's behavior is a property of the loop, not of either level in isolation

### Emergent Properties

Properties that arise from circular causality and cannot be attributed to either level alone include:
- **Robustness**: The macro pattern persists despite turnover of micro components (e.g., cells in a body, employees in an organization)
- **Adaptability**: The system adjusts to perturbations through coordinated micro-level responses organized by macro-level constraints
- **Novelty**: New macro patterns can emerge from changes in micro-level interactions, creating genuinely new capabilities
- **Self-maintenance**: The system sustains itself through the circular loop (autopoiesis)

## Haken's Synergetics

### Order Parameters and Enslaving

Hermann Haken's synergetics (1977, 2004) provides the most developed mathematical framework for circular causality. The key concepts:

**Order parameters**: Macroscopic variables that describe the system's collective behavior. They emerge from microscopic dynamics through a process of self-organization.

**Slaving principle**: Near instabilities (phase transitions), the dynamics of a complex system are governed by a small number of slowly varying order parameters that "enslave" the rapidly varying microscopic degrees of freedom:

```math
\dot{q}_i = f_i(q_1, ..., q_k; p_1, ..., p_n) \quad \text{(fast variables slaved to slow order parameters)}
```

where `q_i` are fast (micro) variables and `p_j` are slow (macro) order parameters.

### The Slaving Principle in Detail

The slaving principle states that near a critical point:

1. The system's dynamics can be decomposed into slow (unstable) modes and fast (stable) modes
2. The fast modes rapidly relax to values determined by the slow modes
3. The slow modes therefore govern the system's long-term behavior
4. The fast modes can be "slaved" (expressed as functions of the slow modes)

```math
q_i(t) \approx h_i(p_1(t), ..., p_k(t)) + \text{fast fluctuations}
```

### Synergetics and the FEP

The connection between synergetics and the FEP is deep:

| Synergetics Concept | FEP Concept |
|--------------------|-------------|
| Order parameter | High-level belief (top of generative hierarchy) |
| Enslaving | Top-down predictions constraining lower levels |
| Control parameter | Environmental context / precision |
| Phase transition | Model switching / perceptual bistability |
| Fluctuations | Noise in Langevin dynamics |
| Instability | High prediction error / model inadequacy |

### The Circular Causality of Synergetics

In synergetics, circular causality is explicit:
1. Microscopic components interact, producing macroscopic order parameters (bottom-up, emergence)
2. Order parameters constrain microscopic components through the slaving principle (top-down, enslaving)
3. The constrained components maintain and modify the order parameters (bottom-up again)

This is exactly the structure of hierarchical predictive coding:
1. Prediction errors from lower levels produce higher-level beliefs (bottom-up)
2. Higher-level beliefs generate predictions that constrain lower-level expectations (top-down)
3. The constrained lower-level processing produces updated prediction errors (bottom-up)

## Order Parameters and the Enslaving Principle

### Examples in Neural Systems

1. **Neural oscillations as order parameters**: Gamma oscillations (30-100 Hz) emerge from interactions among individual neurons but then organize neural firing into phase-locked assemblies. The oscillation enslaves individual neurons to fire at specific phases.

2. **Perceptual states as order parameters**: When viewing an ambiguous figure (like the Necker cube), the perceived interpretation is an order parameter. Individual neural populations are enslaved to maintain one interpretation until a switch occurs.

3. **Emotional states as order parameters**: An emotional state (fear, joy) emerges from distributed neural-bodily interactions but then constrains perception, attention, and behavior. The emotion enslaves lower-level processing to be consistent with the emotional context.

### Mathematical Formalization

Near a bifurcation point, the system's dynamics can be written in normal form:

```math
\dot{p} = \alpha p - \beta p^3 + \sqrt{Q} \xi(t)
```

where `p` is the order parameter, `alpha` is the control parameter (determining whether the ordered state exists), `beta` determines the saturation of the order, and `Q` is the noise amplitude.

For `alpha < 0`: The disordered state (p = 0) is stable (no macroscopic order).
For `alpha > 0`: The ordered states (p = +/- sqrt(alpha/beta)) emerge through symmetry breaking.

### The Enslaving Principle and Free Energy

The slaving of fast variables to slow order parameters can be understood as the fast variables minimizing their conditional free energy:

```math
q_i^* = \arg\min_{q_i} F(q_i | p_1, ..., p_k)
```

Given the current order parameters, the fast variables rapidly relax to the configuration that minimizes free energy. This is precisely what happens in hierarchical predictive coding: lower-level representations rapidly adjust to be consistent with higher-level predictions.

## Applications

### Neuroscience

- **Cortical dynamics**: The circular causality between cortical layers implements predictive processing
- **Brain-body coupling**: Brain states influence bodily states (top-down), bodily states influence brain states (bottom-up), creating the interoceptive loop
- **Consciousness**: May emerge from the strange loop of self-modeling

### Social Systems

- **Institutions and individuals**: Institutions constrain individual behavior (top-down), individuals create and modify institutions (bottom-up)
- **Culture and cognition**: Culture shapes cognitive development (top-down), individual innovations shape culture (bottom-up)
- **Markets and agents**: Market conditions constrain agent behavior, agent behavior creates market conditions

### Ecology

- **Organism and environment**: Organisms adapt to environments (bottom-up), organisms modify environments through niche construction (top-down)
- **Ecosystem and species**: Ecosystem properties constrain species evolution, species interactions create ecosystem properties

### Development

- **Gene regulation**: Gene products regulate other genes (bottom-up networks), cell-level states constrain gene expression (top-down cellular context)
- **Morphogenesis**: Cell-level behavior produces tissue-level patterns, tissue-level patterns constrain cell behavior (see [[biology/morphogenesis|Morphogenesis]])

## Formal Properties

### Fixed Points of Circular Causality

A self-consistent state of circular causality is a fixed point where:

```math
\Phi[\text{micro}^*] = \text{macro}^* \quad \text{and} \quad \Psi[\text{macro}^*] = \text{micro}^*
```

where `Phi` is the bottom-up mapping and `Psi` is the top-down constraint. The system settles into a state where the macro pattern that emerges from the micro dynamics is the same pattern that constrains the micro dynamics.

### Stability Analysis

The stability of circular causal loops depends on the gain of the loop:
- **Gain < 1**: Perturbations decay -- the system returns to its self-consistent state (stable)
- **Gain = 1**: Perturbations persist -- the system is at a critical point (marginally stable)
- **Gain > 1**: Perturbations grow -- the system moves to a new self-consistent state (unstable, leading to phase transition)

### Information Flow in Circular Causality

Information flows in both directions in circular causality:
- **Bottom-up**: Micro-level information is compressed into macro-level descriptions (data compression, sufficient statistics)
- **Top-down**: Macro-level information is elaborated into micro-level predictions (data generation, prediction)

The rate of information flow in each direction determines the relative influence of bottom-up and top-down causation, which is modulated by precision (see [[cognitive/precision_weighting|Precision Weighting]]).

## Key References

- Haken, H. (2004). Synergetics: Introduction and Advanced Topics. Springer.
- Hofstadter, D. R. (1979). Godel, Escher, Bach: An Eternal Golden Braid. Basic Books.
- Edelman, G. M. (1989). The Remembered Present: A Biological Theory of Consciousness. Basic Books.
- Kelso, J. A. S. (1995). Dynamic Patterns: The Self-Organization of Brain and Behavior. MIT Press.
- Friston, K. (2008). Hierarchical models in the brain. PLoS Computational Biology, 4(11), e1000211.
- Thompson, E., & Varela, F. J. (2001). Radical embodiment: Neural dynamics and consciousness. Trends in Cognitive Sciences, 5(10), 418-425.

## Cross-References

- [[systems/emergence|Emergence]] - How macro properties arise from micro dynamics
- [[systems/synergetics|Synergetics]] - Mathematical framework for order parameters and enslaving
- [[cognitive/predictive_coding|Predictive Coding]] - Neural implementation of circular causality
- [[cognitive/hierarchical_inference|Hierarchical Inference]] - Bidirectional inference in cortical hierarchies
- [[cognitive/free_energy_principle|Free Energy Principle]] - Theoretical framework
- [[mathematics/renormalization_group|Renormalization Group]] - Multi-scale organization and coarse-graining
- [[philosophy/enactivism|Enactivism]] - Autonomy through circular self-maintenance
- [[biology/morphogenesis|Morphogenesis]] - Circular causality in biological development
- [[cognitive/precision_weighting|Precision Weighting]] - Modulates balance of top-down and bottom-up
- [[mathematics/markov_blankets|Markov Blankets]] - Formal boundaries in circular causal systems
