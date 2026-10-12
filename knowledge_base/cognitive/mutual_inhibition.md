---
title: Mutual Inhibition
description: Reciprocal suppression between competing neural populations, which produces winner-take-all selection and underlies models of perceptual decision making, rivalry and rhythm generation.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - neural_dynamics
  - decision_making
  - competition
semantic_relations:
  - type: relates
    links:
      - "[[winner_take_all]]"
      - "[[decision_making]]"
      - "[[evidence_accumulation]]"
      - "[[attractor_dynamics]]"
      - "[[binocular_rivalry]]"
---

# Mutual Inhibition

## Overview
Mutual inhibition is an arrangement in which two or more neural populations each suppress the others, directly or through inhibitory interneurons. Activity in one population lowers the input to its competitors, which in turn reduces their ability to suppress it. This positive feedback loop of suppressing the suppressor amplifies small differences between populations and can lead to one population dominating, a form of [[winner_take_all]] selection. Related lateral inhibition between neighbouring units is a standard mechanism for contrast enhancement in sensory systems.

## Dynamics
For two populations with activities $x_1$ and $x_2$ receiving inputs $I_1$ and $I_2$, a simple description is
$$\tau \dot{x}_i = -x_i + f\big(I_i - \beta\, x_j\big),\quad j\neq i,$$
where $\beta$ is the strength of inhibition and $f$ is a non-negative nonlinearity. With weak inhibition the two populations settle to graded activities that reflect their inputs. When inhibition is strong, the symmetric state becomes unstable and the system has two stable states, in each of which one population is active and the other silent. Noise or a small input difference selects which one is reached, and the choice shows hysteresis (see [[attractor_dynamics]]). With slow adaptation added, the dominant state can lose strength and be replaced, producing alternation.

## Functions
- **Decision making.** Leaky competing accumulator models combine accumulation of evidence, decay and mutual inhibition between choice options (Usher and McClelland, 2001). Biophysical circuit models of cortical decision making rely on competing populations with recurrent excitation and shared inhibition (Wang, 2002). Under certain parameter settings, these models approximate optimal sequential evidence accumulation (Bogacz and colleagues, 2006; see [[evidence_accumulation]]).
- **Perceptual rivalry.** Competition between populations representing each eye's input is widely used to model [[binocular_rivalry]].
- **Rhythm generation.** Half-center oscillators, in which two inhibitory populations alternate, are a classic model of locomotor pattern generators.

## Relation to Active Inference
Active Inference treats competition between hypotheses and policies as normalisation of posterior probabilities, usually by a softmax. Process theories propose that lateral inhibition among neuronal populations implements such normalisation and thereby the competition between beliefs. This is a plausible neural implementation proposed for models and not an established finding about specific circuits.

## Related concepts
- [[winner_take_all]] — the selection outcome of strong competition
- [[decision_making]] — the main cognitive application
- [[evidence_accumulation]] — models that combine accumulation with competition
- [[attractor_dynamics]] — the dynamical-systems view of the bistable states
- [[interneurons]] — the inhibitory cells that mediate suppression

## References
- Usher, M., & McClelland, J. L. (2001). The time course of perceptual choice: The leaky, competing accumulator model. Psychological Review, 108, 550-592.
- Wang, X.-J. (2002). Probabilistic decision making by slow reverberation in cortical circuits. Neuron, 36, 955-968.
- Bogacz, R., Brown, E., Moehlis, J., Holmes, P., & Cohen, J. D. (2006). The physics of optimal decision making: A formal analysis of models of performance in two-alternative forced-choice tasks. Psychological Review, 113, 700-765.
