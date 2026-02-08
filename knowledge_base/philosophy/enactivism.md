---
title: Enactivism
type: concept
status: active
created: 2025-02-06
updated: 2025-02-06
tags:
  - enactivism
  - autopoiesis
  - phenomenology
  - sense-making
  - autonomy
  - free-energy-principle
semantic_relations:
  - type: foundation
    links:
      - [[philosophy/4e_cognition|4E Cognition]]
      - [[cognitive/embodied_cognition|Embodied Cognition]]
  - type: relates
    links:
      - [[cognitive/active_inference|Active Inference]]
      - [[cognitive/free_energy_principle|Free Energy Principle]]
      - [[biology/niche_construction|Niche Construction]]
      - [[mathematics/markov_blankets|Markov Blankets]]
  - type: extends
    links:
      - [[cognitive/sensorimotor_coordination|Sensorimotor Coordination]]
      - [[systems/circular_causality|Circular Causality]]
---

# Enactivism

## Overview

Enactivism is a philosophical and scientific approach to cognition that emphasizes the constitutive role of embodied action in mental life. Rooted in the work of Maturana and Varela (1980) on autopoiesis and developed by Varela, Thompson, and Rosch (1991), enactivism proposes that cognition is not the manipulation of internal representations but the enactment of a world through sensorimotor interaction. The relationship between enactivism and the free energy principle (FEP) is complex and contested: some scholars see the FEP as providing the formal framework that enactivism always lacked, while others argue that the FEP's representationalism fundamentally conflicts with enactivist principles. This entry explores both the convergences and the tensions.

## Autopoiesis (Maturana & Varela)

### Definition

Autopoiesis (Greek: auto = self, poiesis = creation) is the property of a system that continuously produces and maintains itself. An autopoietic system is a network of processes that:

1. **Produces its own components**: The system's processes generate the very components that make up the system
2. **Maintains its own boundary**: The system produces and maintains the boundary that distinguishes it from its environment
3. **Is self-referential**: The network of processes refers to itself -- the system's organization is both the product and the producer of its components

### Formal Characterization

An autopoietic system is defined by:
- A topological boundary separating the system from its environment
- A network of component-producing processes within this boundary
- The boundary is itself produced by processes within the system
- The system's organization (the pattern of process relationships) is invariant even as the components are continuously replaced

### Autopoiesis and Markov Blankets

There is a natural correspondence between autopoiesis and the Markov blanket formalism:

| Autopoietic Concept | Markov Blanket Concept |
|---------------------|----------------------|
| Autopoietic boundary | Markov blanket |
| Internal processes | Internal states (mu) |
| Environment | External states (eta) |
| Sensory surface | Sensory states (s) |
| Motor/effector surface | Active states (a) |
| Self-production | NESS maintenance |
| Organizational closure | Free energy minimization |

The key question is whether this correspondence is deep (the FEP formalizes autopoiesis) or superficial (the concepts are structurally similar but philosophically different).

### Minimal Life and Cognition

Maturana and Varela proposed that the minimal criterion for cognition is autopoiesis: any system that maintains its own organization through interaction with an environment is, in the most basic sense, cognitive. This "life-mind continuity thesis" claims that:

```
Life = Autopoiesis = Cognition (at the most basic level)
```

Under the FEP, this corresponds to: any system with a Markov blanket at NESS can be described as performing inference. The FEP thereby provides a formal version of the life-mind continuity thesis.

## Sense-Making

### Definition

Sense-making is the process by which a living system creates meaning in its interactions with the environment. Unlike information processing (which treats meaning as observer-relative), sense-making claims that meaning is intrinsic to the organism-environment coupling:

- The organism does not process pre-given environmental information
- Instead, the organism brings forth a world of significance through its activity
- What counts as relevant, threatening, nourishing, etc., is determined by the organism's own needs and capacities

### Sense-Making Under the FEP

Under the FEP, sense-making can be formalized as the organism's active inference about the environmental states that are relevant to its prior preferences:

```math
\text{Significance}(s) = P(o_{preferred} | s) = \text{how likely this state leads to preferred observations}
```

States that predict preferred outcomes are significant; states that predict non-preferred outcomes are threatening; states that predict no preferred outcomes are irrelevant. The organism's generative model defines a landscape of significance on the environmental state space.

### Value and Valence

Enactivism emphasizes that all cognition is inherently evaluative -- the organism does not first perceive the world neutrally and then evaluate it. Perception is always already imbued with value (valence). Under the FEP, this is captured by the fact that observations are always evaluated relative to prior preferences:

```math
F = D_{KL}[Q(s) || P(s)] + \mathbb{E}_Q[-\ln P(o | s)] + \underbrace{D_{KL}[Q(o) || C(o)]}_{\text{pragmatic value (preferences)}}
```

The prior preference term `C(o)` ensures that every observation is simultaneously perceived and evaluated. There is no perception without preference, just as the enactivists claim.

## Participatory Sense-Making

### Social Enactivism

De Jaegher and Di Paolo (2007) extended the enactivist framework to social cognition through the concept of participatory sense-making: the process by which interacting individuals mutually shape each other's sense-making through their coupled dynamics.

### Key Features

1. **Interaction autonomy**: The interaction itself has a dynamic organization that is not reducible to the individual participants
2. **Mutual modification**: Each participant's sense-making is altered by the interaction
3. **Emergent meaning**: New forms of meaning emerge from the interaction that neither participant could create alone
4. **Breakdown and repair**: The dynamic coupling can break down and be repaired, revealing the interactive nature of understanding

### Participatory Sense-Making and Multi-Agent Active Inference

Under the FEP, participatory sense-making corresponds to multi-agent active inference where:

```math
\dot{\mu}_A = f_A(\mu_A, o_A) \quad \text{where } o_A = g(a_B, s_{shared})
```
```math
\dot{\mu}_B = f_B(\mu_B, o_B) \quad \text{where } o_B = g(a_A, s_{shared})
```

Each agent's observations depend on the other's actions, creating a coupled dynamical system. The emergent interaction dynamics correspond to the collective free energy minimization of the dyad (see [[cognitive/multi_agent_active_inference|Multi-Agent Active Inference]]).

## Lived Body

### The Phenomenological Body

Enactivism draws heavily on phenomenological philosophy, particularly the distinction between:

- **Korper** (the body as object): The body as studied by physiology and anatomy -- a physical thing among other things
- **Leib** (the lived body): The body as experienced from within -- the body as the medium through which we encounter the world

The lived body is not a representation we have of our body but the pre-reflective, experiential ground of all perception and action.

### Lived Body and the Generative Model

Under the FEP, the lived body corresponds to the body schema component of the generative model -- the implicit, dynamically maintained model of the body's configuration and capabilities:

```math
P(o_{proprio}, o_{tactile}, o_{intero} | s_{body}, a)
```

This model is "pre-reflective" in the sense that it operates below the level of conscious access, continuously predicting proprioceptive, tactile, and interoceptive observations without requiring deliberate attention.

### Body-Environment Coupling

The lived body is always already coupled to an environment -- there is no body without a world and no world without a body. Under the FEP, this coupling is formalized through the Markov blanket: the body's sensory and active states constitute the interface through which internal and external states are coupled.

## Phenomenology

### Husserlian Roots

Enactivism draws on Husserl's phenomenology, particularly:
- **Intentionality**: Consciousness is always consciousness of something (directedness toward objects)
- **Constitution**: Objects are constituted through temporal synthesis of experience
- **Horizon structure**: Every experience has a horizon of further possible experiences

### Merleau-Pontian Embodiment

Merleau-Ponty's phenomenology of perception emphasizes:
- **Motor intentionality**: The body's pre-reflective engagement with the world
- **Habit**: Skilled coping that cannot be fully articulated in propositional terms
- **Gestalt perception**: Perception of meaningful wholes, not atomic sense data

### Phenomenology and the FEP

The FEP can be seen as providing formal correlates of phenomenological concepts:

| Phenomenological Concept | FEP Correlate |
|-------------------------|---------------|
| Intentionality | Generative model directed at hidden states |
| Constitution | Variational inference constructing posterior beliefs |
| Horizon | Expected free energy over future states |
| Motor intentionality | Active inference (action as prediction fulfillment) |
| Habit | Optimized policies with high precision |
| Gestalt | Hierarchical generative model with contextual priors |

Whether these correlations amount to genuine formalization or merely analogy is debated.

## Relation to FEP

### Points of Convergence

1. **Organism-environment coupling**: Both frameworks emphasize the constitutive role of organism-environment interaction
2. **Action-perception unity**: Both treat perception and action as aspects of a single process
3. **Embodiment**: Both insist that cognition depends on the body's structure and capabilities
4. **Autonomy**: Both define the organism through its self-maintaining organization
5. **Life-mind continuity**: Both see a deep connection between being alive and being cognitive

### Formal Benefits of the FEP

The FEP offers enactivism:
- **Mathematical precision**: Formal definitions of concepts like autonomy, sense-making, and coupling
- **Quantitative predictions**: Testable predictions about behavior, neural activity, and development
- **Computational models**: Implementable simulations that can be compared with empirical data
- **Unification**: A common framework connecting enactivist ideas to predictive processing, Bayesian inference, and statistical physics

## Critiques of FEP from the Enactive Perspective

### Bruineberg et al.'s Ecological-Enactive Critique

Bruineberg, Kiverstein, and Rietveld (2018) offer an "ecological-enactive" interpretation of the FEP that is sympathetic but cautionary:

1. **Against internalism**: The FEP should not be interpreted as saying that organisms represent or model their environments. Internal states covary with environmental states, but covariance is not representation.

2. **Against optimality**: The FEP should not be interpreted as saying organisms are optimal Bayesian reasoners. Free energy minimization describes a tendency, not an achievement.

3. **For skillful coping**: The FEP is best understood as describing how organisms maintain skilled engagement with their environments, not how they build and update internal world models.

### The Representation Wars

The central philosophical tension is about representations:

**FEP (standard interpretation)**: Internal states encode probability distributions over external states. The generative model is a model of the world. Inference is a form of representation.

**Enactivism**: There are no representations. The organism does not model the world; it is coupled to the world. Internal states are part of the coupling dynamics, not representations of external states.

**Reconciliation attempts**:
- Internal states parametrize distributions but need not be interpreted as "about" external states (deflationary reading)
- The generative model is not stored anywhere; it is the dynamics of the system itself (dynamical reading)
- Representation is a useful description from the observer's perspective but not a property of the system itself (instrumentalist reading)

### Kirchhoff and Froese's Critique

Kirchhoff and Froese (2017) argue that:
- The Markov blanket formalism reintroduces the very boundary between organism and environment that enactivism seeks to dissolve
- The FEP's internal states are functionally isolated from external states (by the blanket), contradicting enactivism's emphasis on direct organism-environment coupling
- The free energy minimization framework is inherently representational, even if one avoids the word "representation"

### Responses

Defenders of the FEP-enactivism synthesis respond that:
- Markov blankets do not imply isolation; they define a conditional independence structure that still allows rich coupling
- The blanket is not a barrier but an interface -- it is the very structure through which coupling occurs
- Free energy minimization can be interpreted non-representationally as a tendency toward self-organization

## Ecological-Enactive Interpretation

### The Skilled Intentionality Framework

Rietveld and Kiverstein (2014) propose a "skilled intentionality framework" that synthesizes ecological psychology, enactivism, and the FEP:

- Organisms are engaged in a field of relevant affordances
- Skilled behavior is the tendency to be responsive to relevant affordances
- The FEP formalizes this as: the organism minimizes free energy by maintaining itself in states from which it can act on relevant affordances

### Direct Perception Revisited

Under the ecological-enactive interpretation, the FEP does not posit that organisms infer environmental states from sensory data (indirect perception). Instead, active inference is a way of formalizing direct perception:

- The generative model is not a representation to be consulted but a set of dynamic dispositions to act
- Prediction errors drive action directly, not through the mediation of internal models
- The organism perceives affordances directly because its action dispositions are already tuned to the environment's action possibilities

## Autonomy

### Organizational Autonomy

Enactivism defines autonomy as the property of a system whose organization is self-determined:

```math
\text{Autonomous system} = \text{system whose boundary and internal dynamics are self-producing}
```

This is stronger than mere self-organization (which can be externally driven). An autonomous system generates its own constraints and operates under its own norms.

### Autonomy Under the FEP

Under the FEP, autonomy corresponds to the system's Markov blanket being self-maintained: the system's internal and active states produce the sensory and active states that constitute the blanket, which in turn enables the internal dynamics that produce the blanket states. This is circular causality (see [[systems/circular_causality|Circular Causality]]).

### Normative Autonomy

A key enactivist claim is that autonomous systems have intrinsic norms: the system's own self-maintenance defines what is "good" and "bad" for it. Under the FEP, these norms correspond to the prior preferences encoded in the generative model -- observations that are consistent with the organism's continued existence have high prior probability.

## Adaptivity

### Beyond Autopoiesis

Di Paolo (2005) argued that autopoiesis alone is insufficient for cognition; adaptivity is also required. An adaptive system not only maintains itself but can modify its interactions with the environment in response to changing conditions:

```math
\text{Adaptivity} = \text{capacity to change coupling with environment to maintain viability}
```

### Adaptivity as Active Inference

Under the FEP, adaptivity is naturally captured by the expected free energy (EFE):

```math
G(\pi) = \text{expected future free energy under policy } \pi
```

An adaptive system selects policies that minimize expected future free energy, anticipating environmental changes and adjusting its behavior proactively. This goes beyond simple autopoietic self-maintenance (which could be achieved by a thermostat-like system) to include genuine flexibility and anticipation.

### The Adaptivity-Optimality Spectrum

Enactivists emphasize that organisms are adaptive, not optimal. The FEP is compatible with this: free energy minimization is achieved approximately, under constraints, and with respect to a particular (possibly inaccurate) generative model. Organisms are satisficing, not optimizing -- they do "well enough" to maintain their NESS, not "the best possible" by some external criterion.

## Key References

- Maturana, H. R., & Varela, F. J. (1980). Autopoiesis and Cognition. D. Reidel.
- Varela, F. J., Thompson, E., & Rosch, E. (1991). The Embodied Mind. MIT Press.
- Thompson, E. (2007). Mind in Life: Biology, Phenomenology, and the Sciences of Mind. Harvard University Press.
- Di Paolo, E. A. (2005). Autopoiesis, adaptivity, teleology, agency. Phenomenology and the Cognitive Sciences, 4(4), 429-452.
- De Jaegher, H., & Di Paolo, E. (2007). Participatory sense-making. Phenomenology and the Cognitive Sciences, 6(4), 485-507.
- Bruineberg, J., Kiverstein, J., & Rietveld, E. (2018). The anticipating brain is not a scientist. Synthese, 195(6), 2417-2444.
- Kirchhoff, M., & Froese, T. (2017). Where there is life there is mind. Entropy, 19(4), 169.

## Cross-References

- [[philosophy/4e_cognition|4E Cognition]] - Broader framework including enactivism
- [[cognitive/embodied_cognition|Embodied Cognition]] - FEP account of embodiment
- [[cognitive/active_inference|Active Inference]] - Formal framework for enaction
- [[cognitive/free_energy_principle|Free Energy Principle]] - Theoretical framework
- [[mathematics/markov_blankets|Markov Blankets]] - Formal boundaries corresponding to autopoietic boundaries
- [[biology/niche_construction|Niche Construction]] - Organism-environment co-constitution
- [[systems/circular_causality|Circular Causality]] - Self-referential dynamics in autonomous systems
- [[cognitive/sensorimotor_coordination|Sensorimotor Coordination]] - Action-perception coupling
