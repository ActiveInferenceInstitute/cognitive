---
title: Inverse Reinforcement Learning
description: The problem of inferring the reward function an agent is optimising from observations of its behaviour, with its ambiguity, standard formulations, and parallels to inferring preferences in Active Inference.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - reinforcement_learning
  - preference_inference
  - imitation
semantic_relations:
  - type: relates
    links:
      - "[[reinforcement_learning]]"
      - "[[reward_function]]"
      - "[[preference_learning]]"
      - "[[markov_decision_process]]"
      - "[[C_matrix]]"
wikidata: Q123916004
---

# Inverse Reinforcement Learning

## Overview
Inverse reinforcement learning (IRL) reverses the usual reinforcement learning problem. Rather than finding a good policy for a given reward function, IRL takes demonstrations of behaviour, assumed to be at least approximately optimal, together with a model of the environment, and infers a reward function that explains them. The recovered reward can then be used to predict behaviour in new situations, to transfer a task to a different environment, or to understand an agent's goals. Ng and Russell (2000) formulated the problem for Markov decision processes.

## Ambiguity and formulations
IRL is ill-posed. Many reward functions make the same behaviour optimal, including the degenerate case of a constant reward, for which every policy is optimal. Methods therefore add assumptions:
- **Margin-based**: choose the reward for which the demonstrated policy beats alternatives by the largest margin (Ng & Russell, 2000).
- **Feature matching**: assume a reward linear in features and match the expected feature counts of the expert, as in apprenticeship learning.
- **Maximum entropy**: among distributions over trajectories that match the feature expectations, choose the one with maximal entropy, which models near-optimal behaviour as probabilistically favouring higher reward (Ziebart et al., 2008).
- **Bayesian IRL**: place a prior over reward functions and compute a posterior given the demonstrations, which represents the remaining ambiguity explicitly.

Practical difficulties include repeated solution of the forward planning problem, the need for an environment model, and sensitivity to the assumption of optimality.

## Relation to Active Inference
The relation is a correspondence of problems. An Active Inference agent does not maximise reward but acts under prior preferences over outcomes, encoded in the C matrix, so inferring preferences from observed behaviour is the analogue of IRL. Because Active Inference is a generative model of action, behaviour can be fitted to it by Bayesian model inversion, estimating preference parameters, precisions and priors from choices; this is the usual way the framework is applied to behavioural data. Related inverse planning accounts of social cognition treat observers as inferring others' goals, linking the topic to theory of mind. Equivalence with particular IRL algorithms is not generally claimed.

## Related concepts
- [[reinforcement_learning]] — the forward problem that IRL inverts
- [[reward_function]] — the object being inferred
- [[preference_learning]] — learning objectives from choices or comparisons
- [[C_matrix]] — preferences over outcomes in Active Inference
- [[theory_of_mind]] — inferring others' goals from their actions
- [[imitation]] — a main use of recovered rewards

## References
- Ng, A. Y., & Russell, S. (2000). Algorithms for inverse reinforcement learning. Proceedings of the 17th International Conference on Machine Learning, 663-670.
- Ziebart, B. D., Maas, A., Bagnell, J. A., & Dey, A. K. (2008). Maximum entropy inverse reinforcement learning. Proceedings of the 23rd AAAI Conference on Artificial Intelligence, 1433-1438.
- Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.). MIT Press.
