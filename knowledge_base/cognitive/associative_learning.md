---
title: Associative Learning
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - learning
  - conditioning
  - behavior
aliases:
  - classical_conditioning
semantic_relations:
  - type: relates
    links:
      - "[[active_inference]]"
      - "[[reinforcement_learning]]"
      - "[[prediction_error]]"
wikidata: Q14330970
---

# Associative Learning

## Overview
Associative learning is learning that two events are related: a stimulus predicts another stimulus (classical or Pavlovian conditioning), or a response leads to an outcome (instrumental or operant conditioning). It is among the most widely studied forms of learning and is found across a very wide range of animals.

## Mechanism
In Pavlovian conditioning, a neutral conditioned stimulus (CS) paired with an unconditioned stimulus (US) comes to evoke a conditioned response. Pairing alone is not sufficient: learning depends on the CS being informative about the US. Blocking (Kamin, 1969) shows that a stimulus redundant with an already-predictive one acquires little association. The Rescorla-Wagner model captures this with an error-correction rule:

$$\Delta V_A = \alpha_A \beta \left(\lambda - \sum_X V_X\right)$$

where $V_A$ is the associative strength of stimulus A, $\alpha_A$ and $\beta$ are salience and learning-rate parameters, $\lambda$ the maximum strength the US supports, and the sum runs over all stimuli present. Learning is driven by the discrepancy between what occurred and what was predicted. Temporal-difference learning extends this to predictions that unfold over time, and dopaminergic firing has been interpreted as the corresponding teaching signal (see [[dopamine_system]]). Extinction, spontaneous recovery and latent inhibition show that the learned association is not simply erased and that context and prior exposure matter. In instrumental learning, an action-outcome (goal-directed) association and a stimulus-response (habitual) association can be dissociated behaviorally.

## Role in Active Inference
The error-correction logic of Rescorla-Wagner is a close relative of prediction-error minimization: the organism updates its expectations in proportion to a mismatch between expected and observed outcomes. In active inference, associations are contingencies encoded in the likelihood and transition parameters of a generative model, learned by accumulating Dirichlet counts as outcomes are observed. Salience and learning rate correspond, loosely, to precision and to the existing evidence in those counts. Instrumental learning is described as inference over policies that minimize [[expected_free_energy]], with outcomes preferred a priori taking the role of reward. This reproduces core conditioning phenomena in simulation, but the match is a reinterpretation of well-established models. Dopamine's role is contested between reward-prediction-error and precision interpretations.

## Related concepts
- [[reinforcement_learning]] — formal framework descended from conditioning theory
- [[prediction_error]] — the mismatch signal that drives learning
- [[hebbian_learning]] — a candidate synaptic mechanism for association
- [[implicit_memory]] — conditioned responses are typically expressed implicitly
- [[reward_processing]] — evaluation of the outcomes that are associated
- [[observational_learning]] — associations can also be acquired by observation

## References
- Pavlov, I. P. (1927). *Conditioned Reflexes*.
- Rescorla, R. A., & Wagner, A. R. (1972). A theory of Pavlovian conditioning: variations in the effectiveness of reinforcement and nonreinforcement. In *Classical Conditioning II*.
- Kamin, L. J. (1969). Predictability, surprise, attention, and conditioning. In *Punishment and Aversive Behavior*.
