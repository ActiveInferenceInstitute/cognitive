---
title: Reward Shaping
description: Adding extra reward terms to speed up reinforcement learning, with the potential-based form that provably preserves optimal policies, and its relation to preference specification in Active Inference.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - reinforcement_learning
  - reward_design
  - policy_invariance
semantic_relations:
  - type: relates
    links:
      - "[[reward_function]]"
      - "[[reinforcement_learning]]"
      - "[[value_function]]"
      - "[[markov_decision_process]]"
      - "[[utility_theory]]"
---

# Reward Shaping

## Overview
Reward shaping modifies the reward signal of a reinforcement learning problem so that the learner receives informative feedback sooner. In tasks where the true reward is sparse, such as success only on reaching a distant goal, random exploration may take very long to find any reward. The term comes from Skinner's use of shaping in animal training, where successive approximations to a target behaviour are reinforced. A hazard is that a poorly chosen bonus can change which behaviour is optimal, so that the agent maximises the bonus instead of the intended goal.

## Potential-based shaping
Ng, Harada and Russell (1999) characterised the shaping rewards that leave the optimal policy unchanged. In a Markov decision process with reward $R(s,a,s')$ and discount $\gamma$, define
$$R'(s,a,s')=R(s,a,s')+\gamma\,\Phi(s')-\Phi(s),$$
for a real-valued potential function $\Phi$ on states. Along any trajectory, the added terms telescope: the discounted sum of shaping terms from time $0$ equals $-\Phi(s_0)$ plus a boundary term that vanishes for bounded $\Phi$ in the infinite-horizon discounted case. The shaped return therefore differs from the original return by a quantity that does not depend on the actions taken, and the optimal values satisfy $Q'^{*}(s,a)=Q^{*}(s,a)-\Phi(s)$ and $V'^{*}(s)=V^{*}(s)-\Phi(s)$. Optimal policies are the same. Ng and colleagues also showed that this form is, in a precise sense, necessary: shaping rewards of other forms can change the optimal policy for some environments.

A natural choice is to set $\Phi$ to an estimate of the optimal value function, which turns the shaped problem into one where the learner already knows how promising each state is. Shaping preserves optimality but not learning speed in general, since a poorly chosen $\Phi$ can slow exploration. Non-potential bonuses, such as curiosity or count-based rewards, are sometimes preferred for exploration, with the understanding that they alter the objective.

## Relation to Active Inference
Active Inference does not specify goals through rewards. Preferences are encoded as a prior distribution over observations, often through the C matrix, and reward is a special case of log-preference. Reward shaping thus corresponds to choosing or reshaping those priors. The invariance result is specific to discounted Markov decision processes and does not carry over automatically to expected free energy, which contains an epistemic term, so the correspondence is an analogy to be checked case by case.

## Related concepts
- [[reward_function]] — the signal being modified
- [[reinforcement_learning]] — setting where shaping is applied
- [[value_function]] — shifts by the potential under the invariance result
- [[markov_decision_process]] — the model for which the theorem is proved
- [[utility_theory]] — preference structure behind reward
- [[reward_prediction_error]] — learning signal computed from the shaped reward

## References
- Ng, A. Y., Harada, D., & Russell, S. (1999). Policy invariance under reward transformations: Theory and application to reward shaping. Proceedings of the 16th International Conference on Machine Learning, 278-287.
- Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.). MIT Press.
- Skinner, B. F. (1938). The Behavior of Organisms: An Experimental Analysis. Appleton-Century.
