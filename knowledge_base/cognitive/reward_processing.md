---
title: Reward Processing
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - reward
  - valuation
  - neuroscience
semantic_relations:
  - type: relates
    links:
      - "[[active_inference]]"
      - "[[reinforcement_learning]]"
      - "[[dopamine_system]]"
---

# Reward Processing

## Overview
Reward processing is the set of neural and computational operations by which outcomes that are valuable to an organism are detected, evaluated, learned about, and used to guide behavior. It covers anticipation of reward, its receipt, and the updating of expectations, and it interacts with decision making, motivation, and emotion.

## Mechanism
Reward is not one process. Berridge and Robinson (1998) argued for dissociating at least three components:

- **Wanting** (incentive salience): motivation to pursue a reward, tied to mesolimbic dopamine.
- **Liking** (hedonic impact): the pleasure of consumption, associated with small opioid and endocannabinoid hotspots in nucleus accumbens and ventral pallidum.
- **Learning**: associative and reinforcement learning that updates the value of cues and actions.

Key regions include the ventral tegmental area and substantia nigra pars compacta (dopamine neurons), ventral and dorsal striatum, orbitofrontal and ventromedial prefrontal cortex (value comparison and outcome representation), amygdala, and anterior cingulate cortex. A central computational idea is the reward prediction error: dopamine neuron firing recorded in monkeys resembled the temporal-difference error, increasing for unexpected reward, shifting to reward-predicting cues with learning, and dipping when an expected reward was omitted (Schultz, Dayan and Montague, 1997). Valuation also depends on delay (see [[temporal_discounting]]), effort, risk, and the animal's current state; the same outcome has different value when sated versus hungry.

## Role in Active Inference
Active inference reframes the notion of reward. There is no separate reward function; instead the agent has prior preferences, a probability distribution over outcomes that it expects to occupy, and it selects policies that minimize [[expected_free_energy]], which favors outcomes with high prior probability while also seeking information (Friston et al., 2010 argued that reward-style value functions are not needed to explain action). Reward-like learning then corresponds to learning about outcomes and policies under such preferences, and curiosity-driven behavior falls out of the epistemic term. This is a coherent alternative description, but it is partly a redescription: with suitable choices, a preference over outcomes can reproduce a reward function (the two are related by log-probability), and the explanatory advantage depends on whether the epistemic and precision terms account for data that reward-based models do not. Interpretation of dopamine as reward prediction error versus precision signal remains contested (see [[dopamine_system]]).

## Related concepts
- [[dopamine_system]] — neuromodulator most associated with reward signals
- [[reinforcement_learning]] — computational formalization of reward-driven learning
- [[pragmatic_value]] — the expected-preference counterpart of reward in active inference
- [[temporal_discounting]] — how delay reduces reward value
- [[associative_learning]] — learning which cues predict reward
- [[decision_making]] — how reward guides choice

## References
- Berridge, K. C., & Robinson, T. E. (1998). What is the role of dopamine in reward: hedonic impact, reward learning, or incentive salience? *Brain Research Reviews*.
- Schultz, W., Dayan, P., & Montague, P. R. (1997). A neural substrate of prediction and reward. *Science*.
- Friston, K. J., Daunizeau, J., Kilner, J., & Kiebel, S. J. (2010). Action and behavior: a free-energy formulation. *Biological Cybernetics*.
