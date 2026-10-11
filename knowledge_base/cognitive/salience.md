---
title: Salience
description: The property that makes a stimulus stand out and attract attention, its bottom-up and goal-driven determinants, computational saliency maps, and its formulation as information gain in Active Inference.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - attention
  - saliency
  - epistemic_value
semantic_relations:
  - type: relates
    links:
      - "[[attention]]"
      - "[[exogenous_attention]]"
      - "[[salience_network]]"
      - "[[epistemic_value]]"
      - "[[dopamine_system]]"
---

# Salience

## Overview
Salience is the degree to which an item stands out from its surroundings and captures attention. The term is used for properties of the stimulus, such as a bright patch in a dark scene, and for the significance of an item to the observer, such as a familiar name heard in a noisy room. These senses overlap but are not identical, and much confusion in the literature comes from treating them as one.

## Determinants and models
- **Stimulus-driven salience.** Local contrast in features such as luminance, colour, orientation or motion makes items conspicuous and tends to attract attention involuntarily (see [[exogenous_attention]]).
- **Goal-driven salience.** Relevance to current tasks and goals, and learned associations with reward or threat, modulate what is selected, so that a physically weak stimulus can dominate if it matters.
- **Saliency maps.** Computational models combine feature contrasts across scales into a single topographic map whose peaks predict where attention and gaze go (Itti & Koch, 2001). Such models explain a portion of free-viewing eye movements, and are generally extended with top-down factors.
- **Incentive salience.** In motivation research, a related notion refers to the "wanting" that makes cues for rewards attractive, and is linked to dopamine signalling.

## Relation to Active Inference
Active Inference supplies a principled definition. In the account of visual search as hypothesis testing (Friston et al., 2012), saliency belongs to the epistemic component of expected free energy: locations are salient to the extent that sampling them is expected to resolve uncertainty about hidden causes, that is, to produce a large expected information gain (see [[epistemic_value]] and [[information_gain]]). Related work shows that unexpected, informative events attract human gaze in proportion to a Bayesian measure of surprise (Itti & Baldi, 2009). Precision also has a role, because attention is cast as the optimisation of the precision of prediction errors (see [[precision_weighting]]). The formulation is a derivation within the framework, but how well it outperforms conventional saliency models on behavioural data is still being examined.

## Related concepts
- [[attention]] — the selective process that salience biases
- [[exogenous_attention]] — involuntary capture by conspicuous stimuli
- [[salience_network]] — brain network that detects behaviourally important events
- [[epistemic_value]] — the Active Inference quantity identified with salience
- [[information_gain]] — expected reduction of uncertainty from sampling
- [[dopamine_system]] — neuromodulation linked to incentive salience

## References
- Itti, L., & Koch, C. (2001). Computational modelling of visual attention. Nature Reviews Neuroscience, 2, 194-203.
- Itti, L., & Baldi, P. (2009). Bayesian surprise attracts human attention. Vision Research, 49, 1295-1306.
- Friston, K., Adams, R. A., Perrinet, L., & Breakspear, M. (2012). Perceptions as hypotheses: Saccades as experiments. Frontiers in Psychology, 3, 151.
