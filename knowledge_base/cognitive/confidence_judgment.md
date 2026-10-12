---
title: Confidence Judgment
description: How people evaluate the likelihood that their own decisions or memories are correct, including calibration, overconfidence, metacognitive sensitivity, and Bayesian accounts of confidence.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - metacognition
  - decision_making
  - uncertainty
semantic_relations:
  - type: relates
    links:
      - "[[metacognition]]"
      - "[[uncertainty_estimation]]"
      - "[[precision_weighting]]"
      - "[[feeling_of_knowing]]"
      - "[[signal_detection]]"
---

# Confidence Judgment

## Overview
A confidence judgment is a subjective estimate of the probability that a decision, answer or memory is correct. It is a core object of metacognition research, because it expresses what a person knows about the reliability of their own cognition. Confidence is typically collected as a rating after a choice, or inferred from behaviour such as willingness to wager or to wait for a reward.

## Calibration and sensitivity
Two properties are usually distinguished. **Calibration** concerns whether stated confidence matches objective accuracy: among answers given with 70% confidence, about 70% should be correct. Studies of general-knowledge questions have repeatedly found overconfidence at high confidence levels, and underconfidence on easy items, a pattern known as the hard-easy effect. **Metacognitive sensitivity** concerns whether confidence discriminates correct from incorrect responses, whatever its absolute level. Measures derived from signal detection theory, such as meta-d', separate sensitivity from the bias toward reporting high or low confidence and from first-order task performance.

## Computational accounts
Many models treat confidence as a readout of a decision variable. In evidence-accumulation models it may reflect the balance of evidence at decision time, and in Bayesian models it is the posterior probability that the chosen option is correct, given the internal evidence. Some accounts add that confidence can draw on information not used for the choice itself, such as response time or evidence arriving after the decision, so confidence need not coincide exactly with the posterior. Neural correlates have been reported in prefrontal, parietal and orbitofrontal regions, and in animals, in orbitofrontal neurons.

## Relation to Active Inference
In Active Inference, uncertainty about hidden states and parameters is represented explicitly, and its inverse, precision, determines how strongly prediction errors update beliefs. Confidence in a perceptual or action choice is naturally read as a property of the posterior or of a higher-level belief about precision. This is a theoretical reading. Whether reported confidence maps onto a specific quantity in a generative model remains an empirical question.

## Related concepts
- [[metacognition]] — the wider capacity to monitor one's own cognition
- [[uncertainty_estimation]] — the quantity confidence is meant to track
- [[precision_weighting]] — the Active Inference counterpart of reliability
- [[feeling_of_knowing]] — confidence applied to memory retrieval
- [[judgment_of_learning]] — confidence about future recall
- [[signal_detection]] — the framework behind sensitivity measures

## References
- Lichtenstein, S., Fischhoff, B., & Phillips, L. D. (1982). Calibration of probabilities: The state of the art to 1980. In D. Kahneman, P. Slovic, & A. Tversky (Eds.), Judgment under Uncertainty: Heuristics and Biases (pp. 306-334). Cambridge University Press.
- Fleming, S. M., & Lau, H. C. (2014). How to measure metacognition. Frontiers in Human Neuroscience, 8, 443.
- Fleming, S. M., & Daw, N. D. (2017). Self-evaluation of decision-making: A general Bayesian framework for metacognitive computation. Psychological Review, 124, 91-114.
