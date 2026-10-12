---
title: Optimal Stopping
description: The problem of choosing when to stop observing a random process to maximise expected reward, including the secretary problem, dynamic programming solution, sequential testing and relevance to exploration decisions.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - decision_theory
  - sequential_decision_making
  - stochastic_processes
semantic_relations:
  - type: relates
    links:
      - "[[exploration_exploitation]]"
      - "[[dynamic_programming]]"
      - "[[decision_theory]]"
      - "[[bandit_problems]]"
      - "[[martingale_theory]]"
      - "[[stopping_rules]]"
wikidata: Q7098950
---

# Optimal Stopping

## Overview
Optimal stopping concerns decisions about when to take a particular action in order to maximise an expected payoff or minimise an expected cost, when information arrives sequentially and the decision cannot be undone. A stopping rule is a strategy that decides at each time, using only information available so far, whether to stop or to continue. Formally the stopping time $\tau$ must be adapted to the observation filtration, so that the event $\{\tau\le t\}$ depends only on data up to time $t$.

## Solution by backward induction
With a finite horizon $T$ and reward $g(X_t)$ for stopping at time $t$, the value of the problem satisfies
$$V_t=\max\big\{g(X_t),\;\mathbb{E}[V_{t+1}\mid\mathcal{F}_t]\big\},\qquad V_T=g(X_T).$$
It is optimal to stop the first time that the immediate reward is at least the expected value of continuing. In general the value process is the smallest supermartingale dominating the reward, the Snell envelope. In continuous time the problem becomes a free boundary problem.

## Classic examples
In the secretary problem, $n$ candidates arrive in random order, each is ranked relative to those already seen, and a candidate must be accepted or rejected immediately. The optimal rule rejects roughly the first $n/e$ candidates and then accepts the first one better than all those seen. As $n$ grows its probability of selecting the best candidate tends to $1/e\approx0.368$. Wald's sequential probability ratio test, which stops collecting data when accumulated log-likelihood ratio crosses one of two thresholds, is optimal in the sense of minimising the expected sample size for given error rates. Pricing American options is also an optimal stopping problem.

## Relation to Active Inference
Deciding when to stop exploring and start exploiting is a stopping problem, and the exploration-exploitation trade-off is built into expected free energy through its epistemic and pragmatic terms. In that framework a stopping decision emerges when the expected information gain from further sampling no longer outweighs the preference-based value of acting, rather than from an explicit optimal stopping calculation. Accumulation of evidence to a threshold, as in sequential tests and drift-diffusion models of choice, has been given Active Inference interpretations. The link is thus conceptual, with a formal correspondence available for specific models.

## Related concepts
- [[exploration_exploitation]] — the trade-off that stopping rules resolve
- [[dynamic_programming]] — the backward induction method
- [[bandit_problems]] — sequential choice with unknown payoffs
- [[martingale_theory]] — the probabilistic tools for stopping times
- [[decision_theory]] — the broader framework of choice under uncertainty

## References
- Ferguson, T. S. (1989). Who solved the secretary problem? Statistical Science, 4, 282-289.
- Chow, Y. S., Robbins, H., & Siegmund, D. (1971). Great Expectations: The Theory of Optimal Stopping. Houghton Mifflin.
- Wald, A. (1945). Sequential tests of statistical hypotheses. Annals of Mathematical Statistics, 16, 117-186.
