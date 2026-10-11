---
title: Behavioral Therapy
description: Psychological interventions derived from learning theory, such as exposure and behavioural activation, and computational hypotheses about why contingency changes alter behaviour, described as research rather than advice.
type: concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - clinical
  - learning
  - behaviorism
semantic_relations:
  - type: relates
    links:
      - "[[associative_learning]]"
      - "[[clinical_psychology]]"
      - "[[computational_psychiatry]]"
aliases:
  - exposure_therapy
---

# Behavioral Therapy

## Overview
Behavioral therapy is a family of psychological interventions that apply principles of learning, chiefly classical and operant conditioning, to change problematic behaviour and emotional responses. It grew out of behaviourism, whose operationalist methodology is discussed in [[operationalism]]. This page describes the approach and the hypotheses computational researchers have proposed; it does not offer guidance on treatment.

## Techniques
- **Exposure**: repeated, structured contact with a feared stimulus without the expected harm. Foa and Kozak (1986) described this as the activation and modification of fear structures through new, incompatible information.
- **Systematic desensitisation**: graded exposure paired with a competing response, introduced by Wolpe (1958).
- **Contingency management and operant methods**: altering consequences so that desired behaviour is reinforced (Skinner, 1953).
- **Behavioural activation**: scheduling activities likely to be rewarding.
- **Habit reversal and related skills training**: building competing responses.

Bouton (2004) argued that extinction, the learning underlying exposure, creates new context-dependent inhibitory learning instead of erasing the original association. This explains observed relapse phenomena such as spontaneous recovery, renewal after a context change, and reinstatement.

## Role in Active Inference
Computational psychiatry offers hypotheses, not established mechanisms, about what such interventions change. In a generative-model view, exposure supplies observations that conflict with a prior expectation of harm; repeated prediction errors revise the transition or likelihood parameters through Bayesian learning (see [[bayesian_learning]], [[prediction_error]]). Contextual dependence of extinction corresponds to models in which the contingency is conditioned on a hidden context state (see [[context_effects]]). Reward-based approaches are modelled through changes in preferences and expected outcomes (see [[preference_learning]]). The explanatory fit is suggestive and partly supported by modelling studies, yet the mapping between a therapeutic technique and specific model parameters is not firmly established, and models of this kind do not on their own predict clinical outcomes.

## Related concepts
- [[associative_learning]] — conditioning principles underlying the techniques
- [[cognitive_therapy]] — the complementary approach that targets beliefs
- [[emotion_regulation]] — regulatory processes affected by interventions
- [[anxiety]] — a condition where exposure-based accounts are most developed
- [[reinforcement_learning]] — formal models of operant contingencies

## References
- Skinner, B. F. (1953). Science and Human Behavior. Macmillan.
- Wolpe, J. (1958). Psychotherapy by Reciprocal Inhibition. Stanford University Press.
- Foa, E. B., & Kozak, M. J. (1986). Emotional processing of fear: Exposure to corrective information. Psychological Bulletin, 99, 20-35.
- Bouton, M. E. (2004). Context and behavioral processes in extinction. Learning & Memory, 11, 485-494.
