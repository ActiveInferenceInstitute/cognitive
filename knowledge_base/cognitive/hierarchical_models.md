---
title: Hierarchical Models
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
      - "[[knowledge_base/cognitive/predictive_processing]]"
      - "[[knowledge_base/cognitive/predictive_processing]]"
      - "[[knowledge_base/cognitive/hierarchical_inference]]"
      - "[[knowledge_base/cognitive/temporal_models]]"
---

# Hierarchical Models

## Overview

A hierarchical generative model stacks several levels of hidden causes, where each level generates the quantities that the level below treats as its data or its priors. Hierarchy lets one model represent structure at several scales: edges, objects and scenes in vision; syllables, words and sentences in speech; movements, actions and plans in behaviour. Hierarchical predictive coding is a well-established computational scheme (Rao and Ballard 1999). Its use in Active Inference for deep temporal planning and its neural implementation are active research programmes with mixed empirical support.

## Main Ideas

**Message passing between levels.** Each level sends predictions downward and receives prediction errors upward. A level updates its beliefs to reduce the errors it receives and, symmetrically, errors at the level above arise from how well its predictions explain the level below ([[knowledge_base/cognitive/predictive_processing]], [[knowledge_base/cognitive/prediction_error]]).

**Precision weighting.** Errors are weighted by their expected reliability. A high-precision error at a given level drives strong belief revision there; a low-precision one is discounted. Attention is commonly modelled as the control of this gain ([[knowledge_base/cognitive/precision_weighting]]). Bastos and colleagues (2012) proposed a mapping from these computations onto the layers of the cortical microcircuit; the mapping is a hypothesis with partial support.

**Separation of timescales.** In deep temporal models (Friston et al. 2017) higher levels evolve more slowly and set context for faster levels. A high-level state such as "making tea" constrains which low-level actions have high prior probability, and a low-level sequence completes before the higher-level state advances. This gives a route to long-horizon planning without enumerating every long action sequence, since a policy at the top level is evaluated over a few abstract steps. This is the main practical argument for depth in agents.

**Hierarchy in learning.** Higher-level parameters act as hyperpriors for lower ones, so evidence at one level can reshape expectations at another. This is the formal basis for treating learning rates, precisions and context effects in one scheme.

## Caveats

Depth multiplies design choices: how many levels, what each represents, how states are coarse-grained. These choices are usually set by the modeller, and learning the structure itself is harder than learning parameters ([[knowledge_base/cognitive/state_space_abstraction]]). Cortical anatomy is also not strictly layered, so the correspondence between algorithmic levels and brain areas is approximate.

## In This Repository

Conceptual coverage is in [[knowledge_base/cognitive/hierarchical_inference]], [[knowledge_base/cognitive/hierarchical_processing]], [[knowledge_base/cognitive/temporal_models]], [[docs/research/architectures/hierarchical]] and [[docs/research/active_inference/hierarchical]]. The code does not implement hierarchical inference: the package notes state that the dispatcher does not advertise hierarchical modes, and `InferenceConfig` accepts only the `discrete` policy type. The nearest working piece is the multi-order structure of `ContinuousActiveInference` in `code/Things/Continuous_Generic`, which stacks generalized coordinates of motion (position, velocity, acceleration) at one level, a different construct from a hierarchy of causes.

## References

- Rao, R. P. N., Ballard, D. H. (1999). Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects. *Nature Neuroscience*.
- Bastos, A. M., Usrey, W. M., Adams, R. A., Mangun, G. R., Fries, P., Friston, K. J. (2012). Canonical microcircuits for predictive coding. *Neuron*.
- Friston, K. J., Rosch, R., Parr, T., Price, C., Bowman, H. (2017). Deep temporal models and active inference. *Neuroscience and Biobehavioral Reviews*.

## Related Resources

- [[knowledge_base/cognitive/predictive_processing]]
- [[knowledge_base/cognitive/predictive_processing]]
