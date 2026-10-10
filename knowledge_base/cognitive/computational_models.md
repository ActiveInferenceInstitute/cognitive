---
title: Computational Models of Cognition
type: knowledge_base
status: stable
created: 2026-02-07
updated: 2026-10-09
tags:
  - cognitive
  - active_inference
aliases:
  - computational_theory
semantic_relations:
  - type: relates_to
    links:
      - "[[knowledge_base/cognitive/active_inference]]"
      - "[[knowledge_base/cognitive/computational_neuroscience]]"
      - "[[knowledge_base/cognitive/bayesian_brain]]"
      - "[[knowledge_base/cognitive/motor_control]]"
      - "[[knowledge_base/cognitive/decision_making]]"
---

# Computational Models of Cognition

## Overview

A computational model of cognition states, as an algorithm, how a mind turns inputs into representations and behaviour, so that the account can be simulated and compared with data. Models are typically described at Marr's computational, algorithmic and implementational levels. Active Inference sits mainly at the first two: it says what problem is being solved (minimise free energy under a generative model) and proposes message-passing schemes that solve it, with neural implementations offered as hypotheses ([[knowledge_base/cognitive/computational_neuroscience]]).

## Three Model Families

**Perception.** Bayesian models treat perception as inference: a prior over causes is combined with a likelihood of the sensory data to give a posterior. Cue combination, illusions and perceptual priors have long been modelled this way, and that part is well established ([[knowledge_base/cognitive/bayesian_brain]], [[knowledge_base/cognitive/perceptual_inference]]). Active Inference adds the claim that the same posterior is computed by minimising variational free energy, which makes the update an approximate-inference scheme that can be run on a computer ([[knowledge_base/mathematics/variational_free_energy]]).

**Decision-making.** Classical accounts maximise expected utility or learn value functions. In Active Inference the agent assigns probability to policies using expected free energy, so goals appear as preferred observations and curiosity appears as information gain ([[knowledge_base/cognitive/decision_making]], [[knowledge_base/mathematics/expected_free_energy]]). Where this matches reinforcement-learning behaviour and where it differs is still being worked out, and published comparisons are limited to small tasks.

**Motor control.** Optimal feedback control models describe movement as the minimisation of a cost under noisy state estimation. The Active Inference alternative holds that the motor system receives predicted proprioceptive consequences, and spinal reflex arcs fulfil them, so there is no separate command signal. This is an active research programme: it reproduces some reaching and handwriting simulations, but experimental tests that discriminate it from optimal control are not decisive ([[knowledge_base/cognitive/motor_control]]).

## What Models Are For

A model earns its place by making quantitative predictions that can fail: choice probabilities, reaction times, belief trajectories, simulated neural responses. Model comparison weighs accuracy against complexity; see [[knowledge_base/cognitive/model_selection]] and [[knowledge_base/research/concepts/model_comparison]]. Active Inference models are usually fitted by estimating a few free parameters, such as precisions or the sensory mapping, and comparing marginal likelihood approximations across variants.

## In This Repository

- `code/Things/Simple_POMDP` implements a discrete perception-plus-decision loop: each step samples a transition, samples an observation, updates beliefs and scores actions by expected free energy.
- `code/Things/Continuous_Generic` (`ContinuousActiveInference`) performs precision-weighted gradient descent on a free energy over generalized coordinates of motion. It returns a gradient-based signal that plays the role of action, which is a simplified stand-in for a motor loop, not a model of a limb.
- `code/tools/src/models/active_inference` holds the shared base class, the discrete generative model and the inference dispatcher ([[knowledge_base/cognitive/artificial_agents]]).

These are small teaching models, tested for internal consistency, and have not been fitted to behavioural data in this repository.

## Related Resources

- [[knowledge_base/cognitive/active_inference]]
- [[knowledge_base/cognitive/computational_neuroscience]]
