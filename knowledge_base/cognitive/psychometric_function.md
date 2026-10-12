---
title: Psychometric Function
description: The function relating the probability of a response to stimulus intensity, with its threshold, slope and lapse parameters, standard fitting and adaptive procedures, and its link to signal detection theory.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - psychophysics
  - perception
  - statistics
semantic_relations:
  - type: relates
    links:
      - "[[psychophysics]]"
      - "[[signal_detection]]"
      - "[[evidence_accumulation]]"
      - "[[reaction_time]]"
      - "[[bayesian_inference]]"
---

# Psychometric Function

## Overview
The psychometric function describes how the probability of a particular response depends on the strength of a stimulus. For a detection or discrimination task it is typically an S-shaped curve that rises from chance level at weak stimulus values to near-perfect performance at strong ones. It is the main object of study in [[psychophysics]], because it summarises sensitivity in a way that is independent of any single trial.

## Parametrisation
A common form is
$$\psi(x) = \gamma + (1-\gamma-\lambda)\,F(x;\alpha,\beta),$$
where $x$ is the stimulus intensity, $F$ is a sigmoid with values from 0 to 1, $\gamma$ is the guess rate (the performance at zero signal, 0.5 in a two-alternative forced-choice task), and $\lambda$ is the lapse rate that accounts for stimulus-independent errors (Wichmann and Hill, 2001). The location parameter $\alpha$ sets the threshold, defined as the intensity giving a criterion level of performance such as 75% correct in a two-alternative task, and the scale parameter $\beta$ sets the slope. Typical choices for $F$ are the cumulative Gaussian, the logistic and the Weibull.

## Estimation
Parameters are usually estimated by maximum likelihood or Bayesian methods from the number of correct responses at each stimulus level. Adaptive procedures, such as staircases and the Bayesian QUEST method (Watson and Pelli, 1983), choose each stimulus level from the responses so far, which concentrates trials near the threshold.

## Link to signal detection theory
Signal detection theory ([[signal_detection]]) gives the function a mechanistic interpretation. If internal responses to signal and noise are Gaussian with equal variance, then performance in a two-alternative forced-choice task is $P(\text{correct})=\Phi(d'/\sqrt{2})$, where $\Phi$ is the standard normal distribution function and $d'$ is the sensitivity index. As the stimulus strengthens, $d'$ grows, and the psychometric function follows. The slope then reflects how quickly sensitivity increases with intensity, and the extent of internal noise.

## Relation to Active Inference
The relation is that of a measurement tool for models that include an observer. A perceptual decision can be modelled as Bayesian inference about the stimulus from noisy sensory evidence, where the width of the likelihood sets the slope of the function and a prior shifts its position (see [[bayesian_inference]]). Active Inference models of perception are compatible with this view, but the psychometric function itself is a descriptive tool that does not depend on the framework.

## Related concepts
- [[psychophysics]] — the study of relations between stimuli and perception
- [[signal_detection]] — the theory that links the function to internal noise
- [[evidence_accumulation]] — models that explain both accuracy and response time
- [[reaction_time]] — the speed measure that accompanies accuracy
- [[bayesian_inference]] — a framework for observer models

## References
- Watson, A. B., & Pelli, D. G. (1983). QUEST: A Bayesian adaptive psychometric method. Perception & Psychophysics, 33, 113-120.
- Wichmann, F. A., & Hill, N. J. (2001). The psychometric function: I. Fitting, sampling, and goodness of fit. Perception & Psychophysics, 63, 1293-1313.
- Kingdom, F. A. A., & Prins, N. (2016). Psychophysics: A Practical Introduction (2nd ed.). Academic Press.
