---
title: 4E Cognition
type: concept
status: active
created: 2025-02-06
updated: 2025-02-06
tags:
  - 4e-cognition
  - embodied
  - embedded
  - enacted
  - extended
  - free-energy-principle
semantic_relations:
  - type: foundation
    links:
      - [[philosophy/enactivism|Enactivism]]
      - [[cognitive/embodied_cognition|Embodied Cognition]]
  - type: relates
    links:
      - [[cognitive/active_inference|Active Inference]]
      - [[cognitive/free_energy_principle|Free Energy Principle]]
      - [[mathematics/markov_blankets|Markov Blankets]]
      - [[biology/niche_construction|Niche Construction]]
  - type: extends
    links:
      - [[cognitive/sensorimotor_coordination|Sensorimotor Coordination]]
      - [[cognitive/social_cognition|Social Cognition]]
---

# 4E Cognition

## Overview

4E cognition is a family of approaches to cognitive science that challenges the classical view of cognition as computation over internal representations in an isolated brain. The "4E" stands for Embodied, Embedded, Enacted, and Extended -- four related but distinct claims about the relationship between mind, body, and environment. The free energy principle (FEP) provides a unifying formal framework for 4E cognition, showing how each of the four "E"s emerges naturally from the mathematics of self-organizing systems with Markov blankets. This entry examines each E in detail and shows how the FEP reconciles them under a single theoretical umbrella.

## Embodied: The Body Shapes the Mind

### The Embodiment Thesis

Embodied cognition holds that cognitive processes are deeply shaped by the body's morphology, sensory apparatus, motor capabilities, and physiological states. The body is not merely a vehicle for the brain but a constitutive part of the cognitive system.

### Core Claims

1. **Sensory embodiment**: The structure of sense organs determines what can be perceived. The human visual system's sensitivity to electromagnetic radiation between 380-700nm, the basilar membrane's tonotopic organization, the skin's distribution of mechanoreceptors -- all these bodily structures constrain the organism's generative model.

2. **Motor embodiment**: The body's action capabilities define the space of possible interventions on the world. A creature that can fly has a different action space -- and therefore a different cognitive relationship to its environment -- than a creature that can only crawl.

3. **Physiological embodiment**: Internal bodily states (hunger, fatigue, arousal, temperature) influence cognitive processes. Interoceptive signals are not peripheral noise but constitutive inputs to the cognitive system.

4. **Morphological embodiment**: The body's physical structure performs computational work (morphological computation) that reduces the burden on neural processing.

### FEP Formalization

Under the FEP, embodiment enters through the generative model's observation and action components:

```math
P(o | s, \text{body}) = \text{observation model shaped by sensory apparatus}
```
```math
P(s_{t+1} | s_t, a, \text{body}) = \text{transition model shaped by motor apparatus}
```

The body determines what the agent can observe and how it can act, which in turn determines the structure of the generative model and the nature of the inference problem.

### Implications

- Different bodies generate different generative models, even in the same environment
- Embodied cognition is not an optional feature but a necessary consequence of being a physical system
- Cognitive abilities cannot be fully understood without reference to the body

## Embedded: The Environment Matters

### The Embeddedness Thesis

Embedded cognition holds that cognitive processes are shaped by and dependent on the physical and social environment in which they occur. Cognition does not happen in a vacuum but is always situated in a specific context.

### Core Claims

1. **Environmental scaffolding**: The environment provides structure that supports cognition. A well-organized workshop supports complex manufacturing; a well-designed interface supports complex decision-making.

2. **Cognitive offloading**: Agents routinely offload cognitive work to the environment. Writing things down reduces memory load; arranging physical objects supports planning; using a calculator offloads arithmetic.

3. **Situated action**: The same agent behaves differently in different environments, not because of different internal states but because different environments afford different actions and provide different scaffolding.

4. **Ecological rationality**: Cognitive heuristics are adapted to specific environmental structures. Fast-and-frugal heuristics work well because they exploit environmental regularities.

### FEP Formalization

Under the FEP, the environment enters through the generative model's prior structure:

```math
P(s) = \text{prior shaped by environmental regularities}
```

The agent's generative model is tuned to the statistical structure of its ecological niche. Changing the environment changes the free energy landscape and therefore the optimal inference strategy.

### Ecological Niches and Free Energy

The organism-environment fit determines the baseline free energy:

```math
F_{baseline} = D_{KL}[Q(s) || P(s | \text{niche})]
```

An organism well-adapted to its niche has low baseline free energy. Removing the organism from its niche (placing a fish out of water, metaphorically or literally) increases free energy dramatically.

## Enacted: Cognition Through Action

### The Enaction Thesis

Enacted cognition (enactivism) holds that cognition is constituted by sensorimotor interaction with the environment. Cognitive states are not internal representations triggered by stimuli but patterns of engagement with the world.

### Core Claims

1. **Perception is active**: Perceiving the world requires actively exploring it -- moving the eyes, turning the head, reaching and touching. Perception is not passive reception but active sampling.

2. **Cognition is action-oriented**: Cognitive processes exist to guide action, and they are structured by the action possibilities available to the agent.

3. **Knowledge is know-how**: Much of what organisms "know" is not propositional knowledge (knowledge-that) but practical skill (knowledge-how). Riding a bicycle involves knowing how to balance, steer, and pedal, not knowing the physics of angular momentum.

4. **World-making**: The organism does not perceive a pre-given world but brings forth a world of significance through its patterns of activity.

### FEP Formalization

Under the FEP, enaction is captured by the dual role of active inference:

**Perception (updating beliefs to match world)**:
```math
\mu^* = \arg\min_\mu F(\mu, o) \quad \text{(perceptual inference)}
```

**Action (updating world to match beliefs)**:
```math
a^* = \arg\min_a F(\mu, o(a)) \quad \text{(active inference)}
```

Cognition is the continuous interplay between these two processes -- neither perception without action nor action without perception constitutes cognition on its own.

### Sensorimotor Contingencies

The enacted view emphasizes sensorimotor contingencies (SMCs) -- the lawful relationships between actions and their sensory consequences. Under the FEP, SMCs are encoded in the generative model's transition function:

```math
P(o_{t+1} | o_t, a_t, s_t)
```

Mastering an SMC means having an accurate generative model of action-conditional sensory changes. This is what it means to know how to see, hear, or touch.

## Extended: Tools Extend the Mind

### The Extended Mind Thesis

Clark and Chalmers (1998) proposed that cognitive processes can extend beyond the brain and body to include external tools and structures. If an external resource plays the same functional role as an internal cognitive process, it is part of the cognitive system.

### The Parity Principle

The core argument (the "parity principle"):

"If, as we confront some task, a part of the world functions as a process which, were it done in the head, we would have no hesitation in recognizing as part of the cognitive process, then that part of the world is (for that time) part of the cognitive process."

### Examples of Extended Cognition

1. **Otto's notebook**: A person with Alzheimer's uses a notebook to store information that a healthy person would store in biological memory. The notebook plays the same functional role as memory.

2. **Smartphone as extended mind**: GPS navigation extends spatial cognition; contacts lists extend social memory; search engines extend general knowledge.

3. **Mathematical notation**: Written equations extend mathematical reasoning beyond what can be held in working memory.

4. **Scientific instruments**: Microscopes and telescopes extend perception; computers extend computation; models extend theory.

### FEP Formalization

Under the FEP, cognitive extension occurs when external resources become part of the agent's generative model:

```math
P(o | s_{internal}, s_{external}, a) = \text{model incorporating external resources}
```

The agent's generative model includes predictions about the behavior of external tools, and the agent relies on these tools for inference just as it relies on internal neural processes.

### Markov Blankets and Extension

The Markov blanket formalism provides a principled account of cognitive extension: the cognitive system's boundary is defined by conditional independence, not by the skin or skull. When an external resource becomes functionally coupled to the agent's internal states in a way that creates a new conditional independence structure, the blanket has effectively expanded.

```math
b_{extended} = b_{biological} \cup b_{tool}
```

The tool's interface becomes part of the agent's sensory and active states, and the tool's internal states become functionally internal to the extended cognitive system.

### Limits of Extension

Not all tool use constitutes cognitive extension. The FEP provides criteria for genuine extension:

1. **Reliable coupling**: The tool must be reliably available and reliably function as expected (high precision in the generative model's predictions about the tool)
2. **Automatic endorsement**: Information from the tool must be treated with the same (or similar) trust as internal information
3. **Prior endorsement**: The agent must have incorporated the tool into its action planning (the tool appears in the generative model's policy space)

## How FEP Unifies 4E

### A Single Framework

The FEP provides a formal framework that naturally accommodates all four E's:

| E | FEP Concept | Mathematical Expression |
|---|-------------|------------------------|
| Embodied | Body shapes observation and action models | P(o|s,body), P(s'|s,a,body) |
| Embedded | Environment shapes priors and context | P(s|niche), F(mu|context) |
| Enacted | Action-perception cycle minimizes F | min F through coupled mu and a updates |
| Extended | Blanket boundary expands to include tools | b = b_bio + b_tool |

### The Unification Argument

The FEP unifies 4E cognition because:

1. **Common mathematics**: All four E's emerge from the same variational principle (free energy minimization)
2. **Common mechanism**: Active inference (coupled perception-action) is the single mechanism underlying all four
3. **Common ontology**: Markov blankets provide a scale-free definition of cognitive systems that naturally accommodates embodiment, embedding, enaction, and extension
4. **No privileged boundary**: The FEP does not assume that cognition is brain-bound; the relevant boundary is the Markov blanket, which may or may not coincide with the skull

### Markov Blankets as Boundaries

The Markov blanket formalism dissolves the traditional boundary problem in 4E cognition:

- **Brain-bound cognition**: If the relevant Markov blanket surrounds the brain, cognition is brain-bound
- **Embodied cognition**: If the blanket surrounds the whole body, cognition is embodied
- **Extended cognition**: If the blanket extends to include external tools, cognition is extended
- **Multiple blankets**: Hierarchically nested blankets allow different levels of analysis -- a brain blanket nested within a body blanket nested within a tool blanket

The right level of analysis depends on the phenomenon being studied.

### Active Inference as Enaction

Active inference is the FEP's formalization of enaction. The key features of enaction are captured:

1. **Action and perception are coupled**: Both are gradient descent on the same free energy functional
2. **Cognition is world-engaged**: The agent cannot minimize free energy without interacting with the world
3. **Meaning is enacted**: The significance of environmental states is determined by their relationship to the agent's prior preferences (not by objective properties)
4. **Knowledge is skillful engagement**: An accurate generative model constitutes skilled know-how for interacting with the world

### Generative Models as Embodied

The generative model is not a disembodied algorithm but a physically instantiated dynamical process:
- Its structure reflects the body's sensorimotor apparatus (embodied)
- Its parameters are tuned to the environment's statistics (embedded)
- Its dynamics constitute the perception-action cycle (enacted)
- Its scope can extend to include external resources (extended)

## Cultural Practices as Extended Cognition

### Culture and the Extended Mind

Cultural practices (language, institutions, rituals, technologies) function as collectively maintained extensions of individual cognition:

- **Language**: A shared generative model for coordinating beliefs through communication
- **Institutions**: Collectively maintained structures that scaffold individual decision-making
- **Rituals**: Embodied practices that maintain shared affective states and social bonds
- **Tools and technologies**: Physical devices that extend individual cognitive capacities

### Cultural Scaffolding Under the FEP

Cultural scaffolding reduces individual free energy by:
1. Providing shared priors (reducing uncertainty through common knowledge)
2. Creating predictable social environments (reducing the complexity of social inference)
3. Offloading cognitive work to cultural artifacts and institutions
4. Enabling cumulative epistemic progress (each generation builds on the last)

### The Cultural Markov Blanket

Social groups can be understood as having their own Markov blankets:
- **Internal states**: Shared beliefs, norms, and practices of group members
- **Sensory states**: Information gathering from the external social and physical environment
- **Active states**: Collective actions that modify the group's environment

This provides a formal basis for understanding group cognition and collective intelligence within the 4E framework.

## Tensions and Debates

### Internalism vs. Anti-Representationalism

The deepest tension within 4E cognition is between those who see the FEP as vindicating internal representations (just more distributed ones) and those who see it as dissolving the need for representations entirely.

### The Scope of Extension

How far does cognitive extension go? Some argue for "radical extension" where entire social systems are cognitive; others argue for more conservative boundaries. The Markov blanket formalism allows both interpretations depending on the scale of analysis.

### Individual vs. Collective

4E cognition raises questions about the boundary between individual and collective cognition. If my smartphone is part of my cognitive system, what about the cloud server it connects to? What about the people who created the apps? The FEP provides formal tools (nested blankets) but does not resolve these philosophical questions.

## Key References

- Clark, A., & Chalmers, D. J. (1998). The extended mind. Analysis, 58(1), 7-19.
- Clark, A. (2015). Surfing Uncertainty: Prediction, Action, and the Embodied Mind. Oxford University Press.
- Varela, F. J., Thompson, E., & Rosch, E. (1991). The Embodied Mind. MIT Press.
- Rietveld, E., & Kiverstein, J. (2014). A rich landscape of affordances. Ecological Psychology, 26(4), 325-352.
- Gallagher, S. (2017). Enactivist Interventions: Rethinking the Mind. Oxford University Press.
- Kirchhoff, M., & Kiverstein, J. (2019). Extended Consciousness and Predictive Processing. Routledge.
- Ramstead, M. J. D., Kirchhoff, M. D., & Friston, K. J. (2020). A tale of two densities: Active inference is enactive inference. Adaptive Behavior, 28(4), 225-239.

## Cross-References

- [[philosophy/enactivism|Enactivism]] - Foundational enactive approach
- [[cognitive/embodied_cognition|Embodied Cognition]] - Detailed treatment of embodiment under FEP
- [[cognitive/active_inference|Active Inference]] - Formal mechanism for enaction
- [[cognitive/free_energy_principle|Free Energy Principle]] - Unifying theoretical framework
- [[mathematics/markov_blankets|Markov Blankets]] - Formal boundaries for cognitive systems
- [[biology/niche_construction|Niche Construction]] - Environmental modification as cognitive extension
- [[cognitive/multi_agent_active_inference|Multi-Agent Active Inference]] - Social aspects of 4E cognition
- [[systems/circular_causality|Circular Causality]] - Reciprocal organism-environment dynamics
