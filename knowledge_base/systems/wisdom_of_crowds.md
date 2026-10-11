---
title: Wisdom of Crowds
description: The finding that pooled judgments of many individuals can be more accurate than most members, the statistical conditions behind it, and how social influence can undermine it.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - collective_judgment
  - aggregation
  - social_influence
semantic_relations:
  - type: relates
    links:
      - "[[collective_intelligence]]"
      - "[[opinion_dynamics]]"
      - "[[swarm_intelligence]]"
      - "[[social_learning]]"
      - "[[probability_theory]]"
---

# Wisdom of Crowds

## Overview
The wisdom of crowds is the observation that the aggregate of many individual estimates or votes is often more accurate than the typical individual and sometimes more accurate than any member or expert. The classic anecdote is by Francis Galton, who in 1907 analysed guesses of the weight of an ox at a livestock fair and found that the median guess was close to the true weight. James Surowiecki popularised the idea in a 2004 book that listed diversity of opinion, independence, decentralisation and a means of aggregation as conditions for good collective judgment.

## Statistical basis
Three results explain why aggregation helps.
- **Error averaging.** If individual errors are unbiased and not perfectly correlated, averaging cancels part of them. For $n$ estimates with equal error variance $\sigma^2$ and pairwise correlation $\rho$, the variance of the mean is $\sigma^2\big(\rho+\tfrac{1-\rho}{n}\big)$, which falls with $n$ but not below $\rho\sigma^2$.
- **Diversity prediction theorem.** For the crowd mean $c$ and truth $\theta$, the squared error of the crowd equals the average individual squared error minus the variance of the estimates around $c$, so diversity directly reduces collective error.
- **Condorcet's jury theorem.** If voters choose between two options independently and each is correct with probability above one half, the majority is correct with probability approaching one as the group grows.

## Failure modes
Aggregation fails when errors share a systematic bias or are correlated through common information. Lorenz and colleagues showed experimentally that showing participants the estimates of others can reduce the diversity of the group without improving its accuracy, so social influence undermines the effect. Information cascades, herding and polarisation are related failures.

## Relation to Active Inference
The link is an analogy that has been explored in modelling. Pooling noisy estimates is reminiscent of combining evidence weighted by precision, and treatments of multi-agent Active Inference examine how shared generative models and communication improve joint inference. No part of the framework predicts the crowd effect itself, which is a result of statistics and social structure.

## Related concepts
- [[collective_intelligence]] — the wider phenomenon of groups performing better than individuals
- [[opinion_dynamics]] — models of how influence changes the independence assumed here
- [[swarm_intelligence]] — decentralised aggregation in animal groups
- [[social_learning]] — learning from others, which can help or harm accuracy
- [[bayesian_inference]] — principled combination of independent evidence

## References
- Galton, F. (1907). Vox populi. Nature, 75, 450-451.
- Surowiecki, J. (2004). The Wisdom of Crowds. Doubleday.
- Page, S. E. (2007). The Difference: How the Power of Diversity Creates Better Groups, Firms, Schools, and Societies. Princeton University Press.
- Lorenz, J., Rauhut, H., Schweitzer, F., & Helbing, D. (2011). How social influence can undermine the wisdom of crowd effect. Proceedings of the National Academy of Sciences, 108, 9020-9025.
