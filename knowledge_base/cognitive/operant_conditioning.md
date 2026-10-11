---
title: Operant Conditioning
description: Learning in which the probability of a behaviour changes according to its consequences, covering reinforcement, punishment, schedules, and its relation to reinforcement learning and Active Inference.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - learning
  - behaviour
  - reinforcement
semantic_relations:
  - type: relates
    links:
      - "[[learning_theory]]"
      - "[[associative_learning]]"
      - "[[reinforcement_learning]]"
      - "[[reward_processing]]"
      - "[[operationalism]]"
---

# Operant Conditioning

## Overview
Operant conditioning, also called instrumental conditioning, is a form of learning in which the frequency of a voluntary behaviour changes as a function of its consequences. It contrasts with Pavlovian conditioning, in which a stimulus predicts another stimulus regardless of what the animal does. The approach is rooted in Thorndike's law of effect, which held that responses followed by satisfying outcomes become more likely, and was developed systematically by B. F. Skinner.

## Core concepts
- **Reinforcement** increases the probability of a response. Positive reinforcement adds a desirable outcome, and negative reinforcement removes an aversive one.
- **Punishment** decreases the probability of a response, by adding an aversive outcome or removing a desirable one.
- **Extinction** is the decline in responding when reinforcement stops.
- **Shaping** reinforces successive approximations to a target behaviour.
- **Schedules of reinforcement** specify when responses are rewarded. Fixed-ratio, variable-ratio, fixed-interval and variable-interval schedules produce characteristic response patterns, and intermittent reinforcement tends to make behaviour resistant to extinction.
- **Discriminative stimuli** signal when a response will be reinforced and so bring behaviour under stimulus control.

Skinner's programme treated learning as an observable relation between behaviour and environment, using rate of response as the measure, which is why the topic is linked to [[operationalism]]. Later work has shown that operant behaviour involves representations of outcome value and of action-outcome contingency, and distinguishes goal-directed control from habit (see [[procedural_memory]]).

## Relation to Active Inference
Operant learning is the empirical territory of [[reinforcement_learning]], where agents learn values or policies from reward. Active Inference reframes the problem: instead of a reward function, the agent has prior preferences over outcomes, and policies are selected by expected free energy, which combines expected preference satisfaction with information gain (see [[expected_free_energy]]). Reinforcement then corresponds to updating beliefs about which actions lead to preferred outcomes. Habit formation appears as learning of a prior over policies. The correspondence is a formal reinterpretation with several demonstrations in simulation, and it does not replace the detailed behavioural findings on schedules.

## Related concepts
- [[learning_theory]] — the broader family of learning accounts
- [[associative_learning]] — Pavlovian and instrumental associations
- [[reinforcement_learning]] — computational treatment of reward-driven learning
- [[reward_processing]] — neural signals for reward and prediction error
- [[dopamine_system]] — neuromodulator linked to reinforcement signals

## References
- Thorndike, E. L. (1898). Animal intelligence: An experimental study of the associative processes in animals. Psychological Review Monograph Supplements, 2(4).
- Skinner, B. F. (1938). The behavior of organisms: An experimental analysis. Appleton-Century.
- Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction (2nd ed.). MIT Press.
