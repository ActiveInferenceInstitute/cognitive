---
title: Learning Curves
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - learning
  - performance
  - measurement
semantic_relations:
  - type: relates
    links:
      - "[[active_inference]]"
      - "[[skill_acquisition]]"
      - "[[learning_theory]]"
---

# Learning Curves

## Overview
A learning curve plots performance against experience: for example response time or error against number of practice trials, or reward against episodes. In machine learning the same term also names plots of training and validation error against training-set size or epochs. Curves summarize how quickly learning proceeds and whether it saturates, and they are used to compare conditions, learners, and theories.

## Mechanism
Common functional forms in human skill learning:

- **Power law of practice** (Newell and Rosenbloom, 1981): $T = a + b\,N^{-c}$, where $T$ is time per trial, $N$ the number of practice trials, $a$ an asymptote, and $b,c$ constants. Performance improves quickly at first and then slowly.
- **Exponential**: $T = a + b\,e^{-cN}$. Heathcote, Brown and Mewhort (2000) argued that exponential fits individual learners better than power functions.
- **Sigmoidal / S-shaped**: slow start, rapid middle phase, and plateau, typical when prerequisite components must be assembled.

Measurement pitfalls matter. Averaging across individuals or items can produce a smooth power-like curve even when each learner shows an abrupt step, so group curves may misrepresent individual learning (Gallistel, Fairhurst and Balsam, 2004). Plateaus can reflect a real ceiling, a change in strategy, or insufficient sensitivity of the measure. Learning curves should be distinguished from retention (forgetting) curves, which plot memory against time after learning, as in Ebbinghaus's 1885 work.

In machine learning, a gap between training and validation curves indicates overfitting; both curves converging at high error indicates underfitting or insufficient model capacity.

## Role in Active Inference
Learning curves are a behavioral readout, not a quantity the theory defines, so the connection is as a test of a model. In active inference simulations, learning appears as free energy and prediction error falling across trials while parameters (such as Dirichlet counts) accumulate. Because each additional observation shifts a count-based posterior less, these models naturally produce diminishing returns, qualitatively resembling power-law or exponential improvement. The match to the specific exponents seen in human data has not been established. Free-energy models can also generate S-shaped or abrupt, insight-like transitions when a structural change in the model, such as model expansion or reduction, occurs (see [[structure_learning]]), which is relevant to the averaging problem noted above. Fitting these models to individual learners trial-by-trial is a standard application.

## Related concepts
- [[skill_acquisition]] — the domain where practice curves were studied
- [[learning_theory]] — theories the curves test
- [[prediction_error]] — quantity that decreases along the curve in model simulations
- [[motor_learning]] — frequent source of power-law data
- [[meta_learning]] — changes in the learning curve across tasks
- [[learning_models]] — formal models that generate predicted curves

## References
- Ebbinghaus, H. (1885). *Über das Gedächtnis*.
- Newell, A., & Rosenbloom, P. S. (1981). Mechanisms of skill acquisition and the law of practice. In *Cognitive Skills and Their Acquisition*.
- Heathcote, A., Brown, S., & Mewhort, D. J. K. (2000). The power law repealed: the case for an exponential law of practice. *Psychonomic Bulletin & Review*.
- Gallistel, C. R., Fairhurst, S., & Balsam, P. (2004). The learning curve: implications of a quantitative analysis. *PNAS*.
