---
title: Hierarchical Active Inference
description: Active Inference with multi-level generative models in which higher levels encode slower, more abstract states and supply empirical priors to faster lower levels, in continuous and discrete formulations.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - active_inference
  - hierarchical_models
  - generative_models
semantic_relations:
  - type: relates
    links:
      - "[[hierarchical_models]]"
      - "[[predictive_coding]]"
      - "[[generative_model]]"
      - "[[timescale_separation]]"
      - "[[continuous_discrete_inference]]"
---

# Hierarchical Active Inference

## Overview
Hierarchical active inference uses a generative model with several levels stacked on top of one another. The lowest level generates sensory observations; each higher level generates the causes or states of the level below. Inference and action then minimise free energy over the whole stack. Because each level supplies priors for the next, the arrangement is a form of empirical Bayes: priors are themselves estimated from data rather than fixed.

## Continuous-time formulation
In hierarchical dynamic models (Friston, 2008) each level $i$ has hidden states $x^{(i)}$ and causes $v^{(i)}$, with the causes of the level below generated from the level above:

$$v^{(i-1)} = g(x^{(i)}, v^{(i)}) + z^{(i)}, \qquad \dot{x}^{(i)} = f(x^{(i)}, v^{(i)}) + w^{(i)}.$$

Predictions flow down and prediction errors flow up, which is the message-passing scheme of [[predictive_coding]]. Precision at each level weights how strongly errors there influence inference.

## Discrete-time formulation
In deep temporal models (Friston et al., 2017) the higher level evolves slowly. Its state at one step specifies the initial state or parameters of a whole sequence of lower-level transitions. The lower level runs for several time steps, and its resulting posterior is passed upward as an observation for the higher level. Temporal depth therefore grows with the number of levels, which allows planning over long horizons without enumerating every low-level policy (see [[policy_selection]]). The same structure has been proposed for hierarchical motor control and motivated behaviour, where higher levels set goals and lower levels implement them (Pezzulo et al., 2018).

## Relation to Active Inference
This is a core part of the formalism, not an analogy: hierarchical generative models are standard in both the continuous and discrete treatments, and have been used in simulations of speech, reading, and sequential behaviour. Whether specific cortical hierarchies implement these message-passing schemes is an empirical question under active investigation. Hierarchy also allows [[continuous_discrete_inference]] mixed models, in which a discrete higher level selects among continuous lower-level dynamics.

## Related concepts
- [[hierarchical_models]] — general statistical background
- [[predictive_coding]] — the neuronal message-passing interpretation
- [[timescale_separation]] — why higher levels evolve more slowly
- [[generative_model]] — the object being stacked into levels
- [[policy_selection]] — planning over deeper temporal horizons
- [[active_inference_loop]] — the perception-action cycle repeated at each level

## References
- Friston, K. (2008). Hierarchical models in the brain. PLoS Computational Biology, 4(11), e1000211.
- Friston, K., Rosch, R., Parr, T., Price, C., & Bowman, H. (2017). Deep temporal models and active inference. Neuroscience & Biobehavioral Reviews, 77, 388-402.
- Pezzulo, G., Rigoli, F., & Friston, K. J. (2018). Hierarchical active inference: A theory of motivated control. Trends in Cognitive Sciences, 22, 294-306.
- Parr, T., Pezzulo, G., & Friston, K. J. (2022). Active Inference: The Free Energy Principle in Mind, Brain, and Behavior. MIT Press.
