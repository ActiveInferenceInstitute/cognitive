---
title: Error Signals
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - active_inference
  - prediction_error
  - precision
semantic_relations:
  - type: relates
    links:
      - "[[active_inference]]"
      - "[[prediction_error]]"
      - "[[precision_weighting]]"
---

# Error Signals

## Overview
An error signal is a quantity measuring the mismatch between what a system predicted and what it received, used to drive learning or correction. In Active Inference the central error signal is the precision-weighted prediction error that gradient-descends on variational free energy; other signals (free energy itself, differences between policies' expected free energies) play related roles.

## Core claims
**Continuous-state form.** In predictive coding the sensory error is ε = y − g(μ), the difference between sensation and the prediction from the current belief μ, and beliefs move down the precision-weighted error: dμ/dt ∝ Π ε · ∂g/∂μ. Precision Π scales how much an error is allowed to change beliefs (see [[precision_weighting]]).

**Discrete-state form.** In the discrete scheme errors are not subtractions; they are log-probability differences between the message from the likelihood and the current log belief, and belief updating drives these to zero.

**Distinct signals.**
- *Sensory prediction error*: observation minus predicted observation.
- *Free energy*: the aggregate accuracy-complexity score of the current beliefs.
- *Policy-level signals*: differences in expected free energy across policies. Friston and colleagues (2014) proposed that dopamine reports changes in policy precision, not reward prediction error.
- *Reward prediction error* in RL is a specific temporal-difference quantity with no primitive counterpart in the framework; it can be reconstructed when C is aligned to reward.

## Role in Active Inference
In this repository, `ActiveInferenceModel.update_beliefs` computes `prediction_error` as the mean squared difference between the observation vector and the predicted observation A·q(s). That scalar feeds two things: `calculate_free_energy` (an accuracy term of 0.5 · error plus a KL complexity term relative to the prior) and `update_precision`, which sets precision toward 1/(error + 1e-8) with an exponential moving average at rate beta (0.9 by default). High error lowers precision and low error raises it. This is a transparent heuristic for adapting confidence, not the canonical precision update from the free-energy gradient, and a squared error stands in for the log likelihood of categorical outcomes.

An error signal is informative only with respect to a model: a persistently large error means either the beliefs are wrong or the model is, and the signal alone does not say which.

## Related concepts
- [[prediction_error]] — types and neural implementation of errors
- [[precision_weighting]] — how errors are scaled
- [[predictive_coding]] — the scheme in which errors are explicit variables
- [[error_correction]] — acting on error to restore a target
- [[reinforcement_learning]] — reward prediction error in contrast
- [[free_energy_minimization]] — what error signals serve

## References
- Friston, K. et al. (2014). The anatomy of choice: dopamine and decision-making. *Philosophical Transactions of the Royal Society B*.
