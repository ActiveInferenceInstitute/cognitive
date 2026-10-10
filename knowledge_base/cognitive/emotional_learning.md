---
title: Emotional Learning
description: Emotional learning is the acquisition, extinction and updating of affective associations, exemplified by Pavlovian fear conditioning, and its relation to prediction-error and Dirichlet-count learning accounts.
type: concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - learning
  - emotion
  - conditioning
semantic_relations:
  - type: relates
    links:
      - "[[associative_learning]]"
      - "[[emotional_processing]]"
      - "[[prediction_error]]"
      - "[[reinforcement_learning]]"
---

# Emotional Learning

## Overview
Emotional learning is the process by which neutral stimuli, places or actions come to carry affective significance through experience. The best-studied case is Pavlovian fear conditioning, in which a neutral cue paired with an aversive outcome later elicits defensive responses on its own. Appetitive conditioning, social learning of fears and the learning of emotional associations by observation follow related principles.

## Mechanism
In fear conditioning, a conditioned stimulus (CS) such as a tone is paired with an unconditioned stimulus (US) such as a mild shock. Rodent work implicates the lateral and basolateral amygdala in forming the association, with the central nucleus driving freezing and autonomic output, and the hippocampus supplying context. Extinction, repeated presentation of the CS without the US, reduces responding. Research following Bouton indicates extinction creates new inhibitory learning rather than erasing the original memory, which accounts for spontaneous recovery, renewal in a new context and reinstatement after a reminder.

Trial-by-trial learning is often described by prediction-error rules. In the Rescorla-Wagner model, associative strength changes in proportion to the difference between the outcome obtained and the outcome predicted, so a fully predicted outcome produces no learning. Dopaminergic signals have been related to reward prediction error, and amygdala and striatal circuits have been related to aversive error signals, though the picture for aversive prediction errors is less settled.

## Role in Active Inference
In discrete-state Active Inference, associations are learned by accumulating Dirichlet concentration counts over the likelihood or transition parameters. Adding one count for the observed outcome changes the expected probability by $(\mathbf{1}_k - p)/(N+1)$, where $N$ is the total prior count. This is a delta rule, an error-driven update whose effective learning rate falls as counts accumulate, so Rescorla-Wagner-like learning is reproduced as a special case, as can be confirmed numerically for any count vector. Aversive outcomes can additionally be encoded as strongly dispreferred in the prior preferences. The correspondence is a formal analogy for the update rule, not evidence that the brain uses Dirichlet counts. Extinction as new learning fits a model with context-dependent parameters, but that is a modelling choice.

## Related concepts
- [[associative_learning]] — the broader family of association-forming processes
- [[emotion_circuits]] — amygdala and associated structures
- [[emotional_memory]] — durability of affectively tagged memories
- [[dopamine_system]] — proposed carrier of error signals
- [[hippocampus]] — contextual gating of emotional responses
- [[value_learning]] — learning of outcome value from experience

## References
- Rescorla, R. A., & Wagner, A. R. (1972). A theory of Pavlovian conditioning. In Black & Prokasy (Eds.), Classical Conditioning II. Appleton-Century-Crofts.
- LeDoux, J. E. (2000). Emotion circuits in the brain. Annual Review of Neuroscience.
