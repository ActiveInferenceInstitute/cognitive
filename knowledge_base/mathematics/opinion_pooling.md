---
title: Opinion Pooling
description: Methods for combining the probability judgments of several agents into a single collective distribution, notably the linear and logarithmic pools, with their properties and their relation to precision-weighted averaging.
type: mathematical_concept
status: stable
created: 2026-10-12
updated: 2026-10-12
tags:
  - probability
  - aggregation
  - collective_decision
semantic_relations:
  - type: relates
    links:
      - "[[probability_theory]]"
      - "[[bayesian_inference]]"
      - "[[belief_updating]]"
      - "[[precision_weighting]]"
      - "[[wisdom_of_crowds]]"
      - "[[collective_intelligence]]"
---

# Opinion Pooling

## Overview
Opinion pooling asks how to combine the probability distributions $P_1,\dots,P_n$ that $n$ individuals, experts or sensors hold over the same set of outcomes into a single collective distribution $P=F(P_1,\dots,P_n)$. The function $F$ is called a pooling operator. The problem arises in forecasting, risk analysis and group decision making. Stone (1961) is an early mathematical treatment, Genest and Zidek (1986) give a critical review of the literature, and Dietrich and List (2017) survey the axiomatic approach.

## The two main pools
**Linear pool.** With non-negative weights $w_i$ summing to one,
$$P(A)=\sum_{i=1}^{n} w_i\,P_i(A).$$
It is the arithmetic mean of the distributions. Pooling commutes with marginalisation: the marginal of the pool is the pool of the marginals. It can be multimodal, and it assigns positive probability to any event that some agent with positive weight considers possible. It does not in general commute with Bayesian updating, and it does not preserve probabilistic independence.

**Logarithmic pool.** With the same weights,
$$P(x)\propto\prod_{i=1}^{n}P_i(x)^{w_i}.$$
It is the normalised geometric mean. If all agents update on the same likelihood $L$, then because $\sum_i w_i=1$ the pooled posterior equals the pool of the posteriors, a property called external Bayesianity. It does not commute with marginalisation, and it assigns zero probability to an outcome that any agent with positive weight rules out.

## Pooling as divergence minimisation
Each pool is the minimiser of a weighted sum of Kullback-Leibler divergences:
$$\text{linear: } \arg\min_P\sum_i w_i\,\mathrm{KL}(P_i\,\|\,P),\qquad \text{logarithmic: } \arg\min_P\sum_i w_i\,\mathrm{KL}(P\,\|\,P_i).$$
The two differ in the direction in which the divergence is measured. Both statements follow from setting the derivative of the corresponding Lagrangian to zero.

## A Gaussian example
If $P_i=\mathcal{N}(\mu_i,\sigma_i^2)$, the logarithmic pool is Gaussian with precision $\sum_i w_i/\sigma_i^{2}$ and mean $\sum_i (w_i/\sigma_i^2)\mu_i\big/\sum_i (w_i/\sigma_i^2)$. It is therefore a precision-weighted average, which connects pooling to the combination of noisy estimates by their reliabilities.

## Impossibility results
No pooling operator satisfies every desirable property at once. Dietrich and List (2017) review characterisation and impossibility theorems that show, for example, how requirements that the collective opinion depend only on the individuals' opinions about each event, or that it preserve unanimous judgments, restrict the choice of operator. The choice of pool is therefore a modelling decision, and Clemen and Winkler (1999) compare methods used in risk analysis.

## Relation to Active Inference
The link is a structural resemblance. Precision-weighted combination of estimates, central to Active Inference, is what a logarithmic pool of Gaussian beliefs does. In multi-agent settings, agents that share or exchange beliefs face a pooling problem. However, nothing in the Active Inference framework prescribes one pooling rule, and the use of a particular pool in a model is a design choice.

## Related concepts
- [[probability_theory]] — the distributions being combined
- [[bayesian_inference]] — updating, which pooling rules may or may not commute with
- [[belief_updating]] — revising beliefs on new evidence
- [[precision_weighting]] — combining estimates by their reliability
- [[wisdom_of_crowds]] — accuracy of aggregated judgments
- [[collective_intelligence]] — group-level cognition

## References
- Stone, M. (1961). The opinion pool. The Annals of Mathematical Statistics, 32(4), 1339-1342. https://doi.org/10.1214/aoms/1177704873
- Genest, C., & Zidek, J. V. (1986). Combining probability distributions: A critique and an annotated bibliography. Statistical Science, 1(1). https://doi.org/10.1214/ss/1177013825
- Clemen, R. T., & Winkler, R. L. (1999). Combining probability distributions from experts in risk analysis. Risk Analysis, 19(2), 187-203. https://doi.org/10.1111/j.1539-6924.1999.tb00399.x
- Dietrich, F., & List, C. (2017). Probabilistic opinion pooling. In A. Hajek & C. Hitchcock (Eds.), The Oxford Handbook of Probability and Philosophy. Oxford University Press. https://doi.org/10.1093/oxfordhb/9780199607617.013.37
