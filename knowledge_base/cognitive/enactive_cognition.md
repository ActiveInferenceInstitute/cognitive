---
title: Enactive Cognition
type: knowledge_base
status: stable
created: 2026-02-07
updated: 2026-10-09
tags:
  - cognitive
  - active_inference
semantic_relations:
  - type: relates_to
    links:
      - "[[knowledge_base/cognitive/embodied_cognition]]"
      - "[[knowledge_base/cognitive/active_inference]]"
      - "[[knowledge_base/philosophy/enactivism]]"
      - "[[knowledge_base/philosophy/4e_cognition]]"
---

# Enactive Cognition

## Overview

Enactive cognition is an approach to cognitive science emphasizing that cognition arises through dynamic sensorimotor coupling between an organism and its environment. Within the Active Inference framework, enactivism maps naturally to the action-perception loop: organisms do not passively receive and process information but actively generate sensory experience through embodied exploration, minimizing variational free energy through the interplay of perception and action.

That sentence states a point of contact, not a consensus. Whether Active Inference is an enactive theory is an open philosophical debate, summarised below.

## Main Ideas of Enactivism

The programme was set out by Varela, Thompson and Rosch in *The Embodied Mind* (1991), building on Maturana and Varela's notion of autopoiesis: a living system continually produces the components that constitute it. Central claims are:

- **Sense-making.** An organism does not register a pre-given world; it brings forth a world of significance through the way its activity serves its self-maintenance.
- **Sensorimotor coupling.** Perception is skilled activity, constituted by how sensory change depends on movement, not an internal picture.
- **Life-mind continuity.** Cognition is an extension of the basic regulation that living systems perform.
- **Participatory sense-making.** Social understanding emerges in the coordinated interaction of two agents (De Jaegher and Di Paolo 2007).

## Relation to Active Inference

Points of agreement are real. Both frameworks stress action, closed loops with the environment, and the continuity of life and mind: the free-energy principle begins from a self-organising system that maintains itself within a bounded set of states ([[knowledge_base/mathematics/markov_blanket]]). Ramstead, Kirchhoff and Friston (2020) argue that Active Inference is enactive in this sense, and Kirchhoff and Froese (2017) defend a strong life-mind continuity from related premises.

Points of disagreement are equally real. Active Inference is commonly written in terms of an internal generative model that represents hidden causes, which many enactivists reject as a hold-over of representationalism. Bruineberg, Kiverstein and Rietveld (2018) separate a view in which the brain builds a model of the world from one in which the organism's whole embodied, environmentally situated dynamics is the model, and argue the second suits the ecological-enactive tradition. Whether "generative model" is a literal posit or a modelling convenience remains contested ([[knowledge_base/philosophy/philosophy_of_mind]]). Autopoietic enactivists also hold that sense-making needs precariousness and metabolism, which a program running on a computer lacks.

## In This Repository

The philosophical pages are [[knowledge_base/philosophy/enactivism]] and [[knowledge_base/philosophy/4e_cognition]]; related cognitive pages are [[knowledge_base/cognitive/embodied_cognition]], [[knowledge_base/cognitive/sensorimotor_coordination]], [[knowledge_base/cognitive/allostatic_control]] and [[knowledge_base/cognitive/embodied_communication]]. Computationally, the closed action-perception loop appears in `Simple_POMDP.step`, where an action changes the hidden state, the state generates an observation, and the observation updates beliefs. That loop is enactive only in the weak sense of coupling; the agent has an explicit probabilistic model, so it is not an autopoietic system.

## References

- Varela, F., Thompson, E., Rosch, E. (1991). *The Embodied Mind*. MIT Press.
- Bruineberg, J., Kiverstein, J., Rietveld, E. (2018). The anticipating brain is not a scientist: the free-energy principle from an ecological-enactive perspective. *Synthese*.
- Ramstead, M. J. D., Kirchhoff, M. D., Friston, K. J. (2020). A tale of two densities: active inference is enactive inference. *Adaptive Behavior*.

## Related Resources

- [[knowledge_base/cognitive/embodied_cognition]]
- [[knowledge_base/cognitive/active_inference]]
