---
title: Stopping Rules
description: Rules that determine when to stop gathering evidence and commit to a choice, including stopping times, the sequential probability ratio test, optimal stopping, and drift-diffusion models of decisions.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - decision_making
  - sequential_analysis
  - evidence_accumulation
semantic_relations:
  - type: relates
    links:
      - "[[optimal_stopping]]"
      - "[[evidence_accumulation]]"
      - "[[decision_making]]"
      - "[[sequential_decision_making]]"
      - "[[martingale_theory]]"
---

# Stopping Rules

## Overview
A stopping rule specifies, for a process that reveals information over time, when to stop observing and act. It captures the trade-off between accuracy, which improves with more evidence, and the cost of time or sampling. Stopping rules appear in sequential statistics, in optimal decision theory and in models of how people and animals decide.

## Mathematical form
A stopping time $\tau$ is a random time such that the event $\{\tau\le t\}$ is determined by information available up to time $t$, so the rule cannot use the future. A fixed sample size is the simplest rule, and a data-dependent rule can stop sooner when evidence is clear.

## Sequential probability ratio test
For two hypotheses $H_0$ and $H_1$, Wald's sequential probability ratio test accumulates the log-likelihood ratio $\Lambda_n=\sum_{i\le n}\ln\frac{p(x_i\mid H_1)}{p(x_i\mid H_0)}$ and stops when it leaves an interval $(-b,a)$, accepting $H_1$ at the upper bound and $H_0$ at the lower. Among all tests with the same error probabilities, it minimises the expected number of observations under each hypothesis, a result due to Wald and Wolfowitz. Thresholds set the error rates, and widening them lowers errors at the cost of longer sampling.

## Models in psychology and neuroscience
In continuous time, accumulation of noisy evidence to a threshold is the drift-diffusion model, in which a decision variable drifts at a rate set by the stimulus and diffuses with noise until it hits one of two bounds. It accounts for the shapes of reaction-time distributions and the speed-accuracy trade-off, and activity resembling accumulation has been recorded in parietal and other cortical areas. Optimal stopping problems, such as the secretary problem, ask for rules that maximise expected reward and are solved by dynamic programming.

## Relation to Active Inference
In Active Inference models of evidence accumulation, beliefs about the hidden cause are updated by each observation, and the agent commits to a response when beliefs, or the precision of policy beliefs, are sufficiently settled or when a trial ends. The decision to stop is typically a modelling choice, for example a fixed number of time steps or a threshold on posterior probability, and a general derivation of the optimal stopping time from expected free energy is not standard. Links to the sequential probability ratio test arise because both accumulate log-evidence.

## Related concepts
- [[optimal_stopping]] — choosing the stopping time to maximise reward
- [[evidence_accumulation]] — the process that stopping rules truncate
- [[decision_making]] — the wider field
- [[signal_detection]] — detection under a fixed sample
- [[martingale_theory]] — the likelihood ratio is a martingale under the null
- [[exploration_exploitation]] — a related trade-off between gathering and using information

## References
- Wald, A. (1945). Sequential tests of statistical hypotheses. Annals of Mathematical Statistics, 16, 117-186.
- Ratcliff, R. (1978). A theory of memory retrieval. Psychological Review, 85, 59-108.
- Gold, J. I., & Shadlen, M. N. (2007). The neural basis of decision making. Annual Review of Neuroscience, 30, 535-574.
