---
title: Sequential Sampling
description: Models in which noisy evidence is accumulated over time until a decision bound is reached, covering the sequential probability ratio test, drift diffusion and links to Bayesian belief updating.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - decision_making
  - evidence_accumulation
  - computational_models
semantic_relations:
  - type: relates
    links:
      - "[[evidence_accumulation]]"
      - "[[decision_making]]"
      - "[[bayesian_inference]]"
      - "[[signal_detection]]"
      - "[[sequential_decision_making]]"
---

# Sequential Sampling

## Overview
Sequential sampling models explain decisions as the gradual accumulation of noisy evidence over time. Instead of a single fixed-duration observation, the decision maker samples evidence repeatedly and responds when the accumulated total reaches a threshold, or bound. The models jointly predict choice probabilities and the distribution of response times, and the trade-off between speed and accuracy follows from the position of the bound.

## Formulation
The statistical origin is Wald's (1947) sequential probability ratio test. For two hypotheses $H_1$ and $H_0$, the log likelihood ratio of successive independent observations $x_t$ is summed,
$$\Lambda_t = \sum_{s=1}^{t} \log \frac{p(x_s \mid H_1)}{p(x_s \mid H_0)},$$
and sampling stops when $\Lambda_t$ crosses an upper or lower bound. The test is optimal in the sense that, for given error rates, it needs on average the fewest observations.

The drift diffusion model of Ratcliff (1978) treats the accumulated evidence as a continuous process,
$$dx = v\,dt + s\,dW,$$
with drift rate $v$ set by stimulus strength, noise $s$, and absorbing bounds at $\pm a$. Additional parameters include a starting point (bias) and a non-decision time for encoding and motor execution (Ratcliff & McKoon, 2008). Related families include race models with separate accumulators for each option and leaky competing accumulators, which add decay and mutual inhibition. Bogacz and colleagues (2006) showed that, under suitable conditions, the diffusion model implements the optimal test.

## Neural correlates
Neurons in the lateral intraparietal area and the frontal eye field of monkeys performing motion discrimination show activity that ramps up at a rate related to stimulus strength and that reaches a similar level before the response. This pattern is often interpreted as evidence accumulation, although alternative interpretations of the ramping, including averaging across trials with step-like changes, are still discussed.

## Relation to Active Inference
In a discrete-state model, belief updating about a hidden state adds the log likelihood of each new observation to the log prior, so the log posterior odds change by the same increments as the sequential probability ratio test. Evidence accumulation is therefore a natural special case of Bayesian belief updating, and Active Inference extends it by letting the agent choose which evidence to sample (Friston et al., 2015). The correspondence is exact only under simplifying assumptions, such as independent observations.

## Related concepts
- [[evidence_accumulation]] — the process these models formalise
- [[decision_making]] — the broader field of choice behaviour
- [[signal_detection]] — the fixed-sample counterpart
- [[bayesian_inference]] — the normative basis of the sequential test
- [[exploration_exploitation]] — a trade-off that also concerns when to stop sampling

## References
- Wald, A. (1947). Sequential Analysis. Wiley.
- Ratcliff, R. (1978). A theory of memory retrieval. Psychological Review, 85, 59-108.
- Bogacz, R., Brown, E., Moehlis, J., Holmes, P., & Cohen, J. D. (2006). The physics of optimal decision making: A formal analysis of models of performance in two-alternative forced-choice tasks. Psychological Review, 113, 700-765.
- Ratcliff, R., & McKoon, G. (2008). The diffusion decision model: Theory and data for two-choice decision tasks. Neural Computation, 20, 873-922.
- Friston, K., Rigoli, F., Ognibene, D., Mathys, C., FitzGerald, T., & Pezzulo, G. (2015). Active inference and epistemic value. Cognitive Neuroscience, 6, 187-214.
