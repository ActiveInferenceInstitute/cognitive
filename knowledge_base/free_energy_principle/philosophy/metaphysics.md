---
title: "Metaphysics of the Free Energy Principle"
type: concept
status: stable
created: 2025-02-06
updated: 2025-02-06
tags:
  - free_energy_principle
  - philosophy
  - metaphysics
  - ontology
  - process_philosophy
  - markov_blankets
  - panpsychism
  - physicalism
  - mathematical_platonism
semantic_relations:
  - type: foundation
    links:
      - [[knowledge_base/free_energy_principle/mathematics/core_principle|Core Principle]]
      - [[knowledge_base/free_energy_principle/mathematics/markov_blankets|Markov Blankets]]
      - [[knowledge_base/free_energy_principle/mathematics/information_geometry|Information Geometry]]
  - type: relates
    links:
      - [[mind_body_problem|Mind-Body Problem]]
      - [[epistemology|Epistemology]]
      - [[consciousness_phil|Consciousness]]
      - [[causality|Causality]]
      - [[knowledge_base/free_energy_principle/systems/self_organization|Self-Organization]]
      - [[knowledge_base/free_energy_principle/systems/emergence|Emergence]]
---

# Metaphysics of the Free Energy Principle

## Overview

The Free Energy Principle (FEP) raises profound metaphysical questions that extend beyond its original domain as a theory of brain function. What kind of principle is the FEP -- a physical law, a mathematical theorem, a methodological framework, or a metaphysical thesis about the nature of reality? What does the FEP say about what *exists* and how it is organized? Does the Markov blanket formalism provide a genuine ontology -- a principled account of what constitutes a "thing" -- or is it merely a useful descriptive tool? These questions situate the FEP at the intersection of philosophy of science, process metaphysics, philosophy of mind, and the foundations of physics.

This document examines the metaphysical landscape of the FEP, addressing its ontological commitments, its relationship to process philosophy, the status of Markov blankets as ontological boundaries, and the debates surrounding panpsychism, physicalism, and mathematical platonism that the FEP has ignited.

## Key Positions

### FEP as Metaphysics vs. Methodology

The most fundamental metaphysical question about the FEP is what *kind* of claim it makes:

**Position 1: FEP as Physical Law**
The FEP is a law of nature, analogous to the second law of thermodynamics. It describes a universal regularity: all systems that persist do so by minimizing free energy. This is an empirical claim about the physical world.

```
FEP as law: For all systems S that persist over time,
            S minimizes variational free energy F[q, o]
            This is a contingent truth about the actual world
```

**Position 2: FEP as Analytical Truth**
The FEP is not an empirical discovery but a mathematical consequence of defining "system" in terms of Markov blankets and "persistence" in terms of non-equilibrium steady states. It is true by definition -- a tautology, not an empirical law.

```
FEP as tautology: IF a system has a Markov blanket at NESS,
                  THEN its internal dynamics can be described as free energy minimization
                  This is a necessary truth following from the definitions
```

**Position 3: FEP as Methodological Principle**
The FEP is neither a law nor a tautology but a **research heuristic** -- a way of organizing inquiry. It says: "For any self-organizing system, *try* to describe its dynamics as free energy minimization and see what this reveals." On this view, the FEP does not claim that systems *actually* minimize free energy but that describing them as if they do is scientifically productive.

```
FEP as method: Treat system S as if it minimizes F
               Derive predictions from this assumption
               Test predictions empirically
               The FEP is vindicated if this is more productive than alternatives
```

**Resolution Attempt (Friston, 2019)**:
Friston's "particular physics" paper attempts to resolve this debate by deriving the FEP from the assumption that a system possesses a Markov blanket at NESS. If these conditions hold, free energy minimization follows necessarily. The FEP is therefore:
- **Analytic** relative to the definitions of Markov blanket and NESS
- **Empirical** in that whether a given system satisfies these conditions is an empirical matter
- **Methodological** in that identifying the Markov blanket and generative model is a modeling choice

This position is analogous to the status of the principle of least action in physics: it is mathematically derivable from the equations of motion (analytic), those equations describe the actual world (empirical), and the variational formulation is often the most illuminating way to set up a problem (methodological).

### Markov Blankets as Ontological Boundaries

Perhaps the most metaphysically significant aspect of the FEP is its claim that **Markov blankets define the boundaries of things**. This is a thesis about ontology -- about what constitutes an individual entity:

```
Traditional ontology: Things exist first; boundaries are derived from things
FEP ontology: Boundaries (Markov blankets) come first; things are derived from boundaries

A "thing" = a set of states {mu} such that there exists a set of states {b}
            forming a Markov blanket that statistically separates {mu}
            from external states {s}
```

This is a radical proposal. Under the FEP, "thing-ness" is not a primitive metaphysical category but is constituted by a particular statistical structure -- conditional independence mediated by blanket states. A thing is anything that can be separated from its environment by a Markov blanket.

**Consequences**:

1. **Things are scale-relative**: Markov blankets exist at every scale -- cells, organs, organisms, ecosystems, societies. What counts as a "thing" depends on the scale at which one identifies blankets.

2. **Things are dynamic**: Markov blankets are not static walls but dynamic statistical boundaries. The "boundary" of an organism is constantly changing as molecules are exchanged with the environment.

3. **Things are nested**: Markov blankets contain sub-blankets. An organism is a blanket of blankets of blankets. The hierarchy of thing-ness is potentially unlimited.

```
Nested blankets:
  Organelle MB ⊂ Cell MB ⊂ Tissue MB ⊂ Organ MB ⊂ Organism MB ⊂ Social group MB

Each level constitutes a "thing" at that scale
No single level is ontologically privileged
```

**Criticism (Bruineberg et al., 2022; Colombo & Palacios, 2021)**:

Several philosophers have challenged the ontological interpretation of Markov blankets:

- **Observer-dependence**: The identification of a Markov blanket depends on the modeler's choice of partition. Different partitions yield different blankets. If blankets are observer-dependent, they cannot ground an observer-independent ontology.

- **Temporal ambiguity**: Markov blankets in the FEP literature are identified at the non-equilibrium steady state. But real systems are rarely at exact NESS. The blanket is an idealization, not a precise boundary.

- **The blanket-as-pearl problem**: For any set of variables, a Markov blanket can in principle be identified. Does every set of variables constitute a "thing"? If so, the ontological claim is trivial.

### Process Ontology (Whitehead Connections)

The FEP resonates deeply with **process philosophy** -- the metaphysical tradition, associated primarily with Alfred North Whitehead (1929), that holds that reality consists fundamentally of *processes* rather than *things*:

```
Substance ontology (Aristotle, Descartes):
  The world is composed of substances (things) with properties
  Change = alteration of properties of persistent substances
  Process is derivative: things come first, processes are what things do

Process ontology (Whitehead, FEP):
  The world is composed of processes (events, flows, dynamics)
  "Things" = stable patterns within processes
  Substance is derivative: processes come first, things are stable process-patterns
```

The FEP aligns with process ontology in several key respects:

1. **Dynamics are fundamental**: The FEP describes flows (stochastic differential equations), not static entities. An organism IS its dynamics -- the characteristic flow through state space.

2. **Things are steady states**: Under the FEP, a "thing" is a pattern of dynamics that maintains itself at a non-equilibrium steady state. It is a process that has achieved sufficient stability to persist, not a substance that happens to be dynamic.

3. **Becoming over being**: The solenoidal (non-equilibrium) flow ensures that even at steady state, the system is in constant motion. There is no static "being" -- only dynamic "becoming" that happens to maintain a stable statistical structure.

4. **Relations are primary**: The Markov blanket defines a thing through its *relations* (conditional independence structure), not through intrinsic properties. This is a relational, structural ontology.

**Whitehead's Actual Occasions and FEP**:

Whitehead proposed that reality consists of "actual occasions of experience" -- momentary events that "prehend" (take in) data from past occasions and creatively synthesize a new unity:

```
Whitehead's actual occasion:
  Physical pole: Reception of data from past occasions (= sensory input)
  Mental pole: Creative response to that data (= inference and action)
  Satisfaction: The completed occasion becomes data for future occasions (= steady state)

FEP agent at each time step:
  Sensory input: o_t (= physical pole)
  Inference and action: mu_t -> a_t (= mental pole)
  State at t+1: x_t+1 (= satisfaction, data for next step)
```

The structural parallel is striking. Both Whitehead and the FEP propose that every entity, at every scale, has a receptive (physical/sensory) aspect and a responsive (mental/active) aspect. The FEP may be understood as a mathematical formalization of Whitehead's process ontology.

## Detailed Discussion

### Panprotopsychism and Panpsychism

The FEP's claim that any system with a Markov blanket performs "inference" raises the question of **panpsychism** -- the view that mentality is ubiquitous in nature:

```
FEP claim: Any system with a Markov blanket at NESS can be described
           as performing approximate Bayesian inference

Panpsychist reading: Since "inference" is a mental property,
                     and Markov blankets are ubiquitous,
                     mentality is ubiquitous

Conservative reading: "Inference" as used in the FEP is a mathematical description
                      of dynamics, not a literal attribution of mental properties
```

The debate turns on whether the FEP's "as if" inference constitutes genuine mentality:

**Strong panpsychism**: Every system with a Markov blanket is, to some degree, a subject of experience. Rocks, thermostats, and galaxies have primitive forms of experience. This is the most literal reading of the FEP.

**Panprotopsychism** (Chalmers, 2015): The FEP describes *proto-mental* properties -- the precursors of mentality -- that are ubiquitous but that constitute full mentality only when combined in the right way (sufficient complexity, integration, temporal depth). This is the position most FEP proponents favor (see [[consciousness_phil|Consciousness]]).

**Deflationary reading** (Colombo, 2021): The FEP's "inference" is a purely mathematical description with no mentalistic implications. Calling a system's dynamics "inference" is a useful metaphor, not a literal attribution. This dissolves the panpsychism worry but raises the question of what, if anything, the FEP says about the mind.

**Kirchhoff and Froese (2017)** advocate a middle path: the FEP supports a "strong life-mind continuity thesis" in which the organizational principles that constitute life are the same as those that constitute mind. This does not attribute full consciousness to all living systems but claims that the distinction between living and mental is one of degree, not kind:

```
Life-mind continuity:
  Minimal life (bacteria): Minimal Markov blanket, simple inference
  Complex life (mammals): Deep hierarchical inference, precision modulation
  Consciousness: Sufficient hierarchical depth + self-modeling + temporal depth

The gap between life and mind is quantitative, not qualitative
```

### Physicalism vs. Neutral Monism

The FEP's dual-aspect character -- describing the same system as both physical dynamics and inferential process -- raises questions about its relationship to physicalism:

**Physicalism**: Everything that exists is physical. Mental properties are identical to, or supervene on, physical properties.

```
Physicalist FEP: The FEP describes physical dynamics
                 The "inference" interpretation is just a useful description
                 of those physical dynamics
                 There is nothing over and above the physics
```

**Neutral monism**: Neither the physical nor the mental is fundamental. Both are aspects of a neutral underlying reality.

```
Neutral monist FEP: The FEP describes a reality that is neither
                    purely physical nor purely mental
                    Physical description: stochastic dynamics, flows, NESS
                    Mental description: inference, beliefs, free energy minimization
                    Neither description is more fundamental
                    Both are aspects of a neutral process
```

Friston's own position appears closest to neutral monism (or dual-aspect monism -- see [[mind_body_problem|Mind-Body Problem]]): the dynamics and the inference are two descriptions of one process, and neither is reducible to the other. The physical equations and the inferential equations are mathematically equivalent -- they describe the same thing in different vocabularies.

This raises a subtle point: if the FEP is correct, the distinction between "physical" and "mental" may be an artifact of our descriptive practices rather than a feature of reality. The same system, described in one vocabulary, is "physical dynamics"; described in another, it is "Bayesian inference." The FEP suggests that these vocabularies are systematically intertranslatable:

```
Physical vocabulary:    dx/dt = f(x) + omega(t)
Inferential vocabulary: dF/dq = 0 at q = q*
Mapping:               f(x) <-> gradient descent on F
                       x <-> sufficient statistics of q
                       omega <-> stochastic fluctuations enabling exploration
```

### The Explanatory Gap Revisited

The metaphysical framing of the FEP sharpens the **explanatory gap** between physical/functional description and phenomenal experience:

```
What the FEP explains:
  - Structure of inference (generative models, hierarchies)
  - Dynamics of inference (free energy minimization, precision optimization)
  - Function of inference (adaptive behavior, homeostasis)

What the FEP does not explain:
  - Why inference is accompanied by experience
  - Why particular inferences have particular qualitative characters
  - The very existence of subjectivity
```

Three metaphysical responses:

1. **Type-A materialism** (deny the gap): The FEP will eventually close the gap. Once we fully understand the inferential dynamics, there will be nothing left to explain. The "hard problem" is a philosophical confusion.

2. **Type-B materialism** (accept the gap, deny its significance): Consciousness is identical to certain inferential processes, but this identity is an a posteriori discovery, like "water = H2O." The gap is epistemic, not ontological.

3. **Property dualism / neutral monism**: The gap reflects a genuine feature of reality. Phenomenal properties are additional to (or a different aspect of) physical/functional properties. The FEP captures the functional side but not the phenomenal side.

### FEP and Mathematical Platonism

The FEP's reliance on mathematical structures (variational calculus, information geometry, Bayesian probability) raises questions about the ontological status of these mathematical objects:

```
Mathematical platonism: Mathematical structures exist independently of minds
                        The FEP describes real mathematical structures
                        that the physical world instantiates

Mathematical nominalism: Mathematical structures are human constructs
                         The FEP uses convenient mathematical fictions
                         to describe physical regularities

Structural realism:     What exists is the structure itself
                        The FEP captures the relational structure of reality
                        whether or not the structure is "mathematical" in a
                        Platonic sense
```

The FEP's universality -- its claim to apply to *any* self-organizing system -- pushes toward a structural realist interpretation. If the same mathematical structure (free energy minimization) is instantiated by neurons, immune cells, social groups, and potentially any persisting system, then this structure seems to be a genuine feature of reality, not an artifact of our description.

This connects to Tegmark's (2014) mathematical universe hypothesis: if the fundamental nature of reality is mathematical structure, then the FEP describes one of the most fundamental structural features of reality -- the principle by which structure maintains itself.

### Universality Claims

The FEP is often presented as a **universal** principle -- applicable to any system that can be said to exist. This universality claim has strong metaphysical implications:

```
Weak universality: The FEP applies to all biological systems
                   (brains, immune systems, cells, ecosystems)

Moderate universality: The FEP applies to all self-organizing systems
                       (biological + some non-biological: crystals, weather patterns)

Strong universality: The FEP applies to anything with a Markov blanket at NESS
                     (potentially everything that persists)
```

Strong universality, if correct, would make the FEP a principle of extraordinary scope -- a principle of existence itself:

```
To exist (as a particular, bounded entity) = to have a Markov blanket
To persist = to maintain that blanket at NESS
To maintain that blanket = to minimize free energy
Therefore: To exist and persist IS to minimize free energy
```

This is a metaphysical claim of the highest order. Critics (Colombo & Palacios, 2021) argue that this universality is purchased at the cost of vacuity: if the FEP applies to everything, it predicts nothing specific. Defenders respond that the FEP provides a *framework* within which specific predictions can be derived for particular systems, just as the principle of least action provides a framework within which specific physical predictions can be derived.

### Ontological Status of Generative Models

A key metaphysical question concerns the ontological status of the generative model:

```
Realism about generative models:
  The generative model p(o, s | theta) is a real structure
  instantiated in the physical organization of the system
  It has genuine representational content

Instrumentalism about generative models:
  The generative model is a useful fiction -- a description
  that helps US (the scientists) understand the system
  The system itself does not "have" a model

Structural realism:
  The system instantiates a structure that is isomorphic to
  a generative model, but whether this counts as a "real model"
  is a matter of convention
```

This debate parallels the broader debate about scientific realism. The FEP does not resolve it but sharpens it by providing a precise formal object (the generative model) whose ontological status can be interrogated.

## Connections to Other Frameworks

### Autopoiesis and Enactivism

The FEP's metaphysics connects to autopoietic theory (Maturana & Varela, 1980), which defines living systems as self-producing networks. Under autopoiesis, the boundary of a living system is self-generated:

```
Autopoiesis: The system produces the components that produce the boundary
             that separates the system from the environment

FEP:         The dynamics maintain the Markov blanket
             that separates internal from external states

Convergence: Both hold that the boundary is self-maintaining and constitutive of the entity
```

The FEP can be seen as a mathematical formalization of autopoiesis: the Markov blanket IS the autopoietic boundary, and free energy minimization IS the self-production that maintains it.

### Information-Theoretic Ontology

The FEP resonates with information-theoretic approaches to ontology (Wheeler's "it from bit," Floridi's informational structural realism):

```
Wheeler: "Every it -- every particle, every field of force, even the
          spacetime continuum itself -- derives its function, its meaning,
          its very existence entirely from binary choices, bits."

FEP:     Every "thing" derives its existence from the information-theoretic
         structure of the Markov blanket -- the conditional independence
         relations that define it as a bounded entity.
```

On this view, information is not merely epistemic (what we know) but ontological (what exists). The FEP suggests that the most fundamental feature of reality is not matter or energy but *statistical structure* -- the pattern of conditional independences that constitutes the world's partition into interacting entities.

### Mereology (Parts and Wholes)

The FEP's nested Markov blankets provide a formal mereology -- a theory of parts and wholes:

```
Classical mereology: Wholes are composed of parts; parts are prior to wholes
FEP mereology:      Wholes and parts are defined simultaneously by the
                    blanket decomposition; neither is ontologically prior

Part = sub-system with its own Markov blanket within the whole
Whole = system whose Markov blanket encompasses the parts
Relation = the blankets nest: part-blanket ⊂ whole-blanket
```

This yields a non-reductive, holistic mereology: the whole is not merely the sum of its parts (because the whole has emergent statistical structure -- its own blanket -- that is not present in any individual part), but neither are the parts mere abstractions from the whole (because each part has its own blanket and its own dynamics).

## Open Questions

1. **Is the FEP falsifiable?** If the FEP is an analytical truth (following from definitions), can it be empirically tested? What would count as evidence against it? This is a question about the FEP's status as a scientific theory.

2. **What is the correct interpretation of "inference"?** Is the FEP's use of mentalistic language (beliefs, inference, models) literal or metaphorical? The answer determines whether the FEP has implications for philosophy of mind or is purely a physical theory.

3. **Ontological priority**: In the FEP ontology, what is fundamental -- the dynamics, the blanket structure, the free energy functional, or the generative model? Each has been treated as primary by different authors.

4. **The combination problem**: If panprotopsychism is correct and proto-mental properties are ubiquitous, how do they combine to form full consciousness? The FEP's nested blankets suggest a compositional answer (complex experience = composed of simpler proto-experiences), but the details are unclear.

5. **Pluralism vs. monism**: Does the FEP support ontological monism (one kind of stuff) or pluralism (multiple kinds)? The dual-aspect interpretation suggests monism; the nested blanket interpretation might support pluralism (different "things" at different scales).

6. **Temporal ontology**: Is the FEP's time symmetric or does it presuppose a direction of time? The NESS formulation involves time-reversal symmetry breaking through the solenoidal flow, but the metaphysical implications of this are unexplored.

7. **FEP and quantum mechanics**: Several authors have noted formal parallels between the FEP and quantum mechanics (both involve variational principles, both feature observer-dependent descriptions). Are these parallels superficial or do they indicate a deeper connection?

## Key References

1. Friston, K. J. (2019). A free energy principle for a particular physics. *arXiv preprint* arXiv:1906.10184.
2. Kirchhoff, M. D., Parr, T., Palacios, E., Friston, K. J., & Kiverstein, J. (2018). The Markov blankets of life: Autonomy, active inference, and the free energy principle. *Journal of the Royal Society Interface*, 15(138), 20170792.
3. Kirchhoff, M. D., & Froese, T. (2017). Where there is life there is mind: In support of a strong life-mind continuity thesis. *Entropy*, 19(4), 169.
4. Ramstead, M. J. D., Badcock, P. B., & Friston, K. J. (2018). Answering Schrodinger's question: A free-energy formulation. *Physics of Life Reviews*, 24, 1-16.
5. Bruineberg, J., Dolega, K., Dewhurst, J., & Baltieri, M. (2022). The Emperor's new Markov blankets. *Behavioral and Brain Sciences*, 45, e183.
6. Colombo, M., & Palacios, P. (2021). Non-equilibrium thermodynamics and the free energy principle in biology. *Biology & Philosophy*, 36(5), 41.
7. Colombo, M. (2021). First principles in the life sciences: The free energy principle, organicism, and mechanism. *Synthese*, 198, 3463-3488.
8. Whitehead, A. N. (1929). *Process and Reality*. Macmillan.
9. Chalmers, D. J. (2015). Panpsychism and panprotopsychism. In T. Alter & Y. Nagasawa (Eds.), *Consciousness in the Physical World: Perspectives on Russellian Monism*. Oxford University Press.
10. Tegmark, M. (2014). *Our Mathematical Universe: My Quest for the Ultimate Nature of Reality*. Knopf.
11. Maturana, H. R., & Varela, F. J. (1980). *Autopoiesis and Cognition: The Realization of the Living*. D. Reidel.
12. Ramstead, M. J. D., Friston, K. J., & Hippolito, I. (2020). Is the free energy principle a formal theory of semantics? *Entropy*, 22(8), 889.

## See Also

- [[mind_body_problem|Mind-Body Problem]] -- Dual-aspect monism and the FEP
- [[epistemology|Epistemology]] -- What can we know under the FEP?
- [[consciousness_phil|Consciousness]] -- Phenomenal experience and the FEP
- [[causality|Causality]] -- Causal structure and the FEP ontology
- [[free_will|Free Will]] -- Agency in a process-ontological framework
- [[knowledge_base/free_energy_principle/mathematics/core_principle|Core Principle]] -- The mathematical foundations
- [[knowledge_base/free_energy_principle/mathematics/markov_blankets|Markov Blankets]] -- Formal definition of blankets
- [[knowledge_base/free_energy_principle/mathematics/information_geometry|Information Geometry]] -- The geometry of inference
- [[knowledge_base/free_energy_principle/systems/self_organization|Self-Organization]] -- How structure maintains itself
- [[knowledge_base/free_energy_principle/systems/emergence|Emergence]] -- Emergent properties and levels
- [[knowledge_base/free_energy_principle/biology/evolution|Evolution]] -- Evolutionary ontology and the FEP
