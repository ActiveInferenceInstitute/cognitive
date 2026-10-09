---
title: Computational Psychiatry
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - psychiatry
  - computational_modeling
  - psychopathology
semantic_relations:
  - type: relates
    links:
      - "[[active_inference]]"
      - "[[psychiatry]]"
      - "[[psychiatric_disorders]]"
      - "[[precision_weighting]]"
---

# Computational Psychiatry

## Overview
Computational psychiatry is a research field that applies formal models of brain and behaviour (reinforcement learning, Bayesian inference, neural network and dynamical-systems models) to mental-health conditions. Its aim is to link neurobiological mechanisms to symptoms through explicit, testable computations, rather than through verbal descriptions or purely statistical classification of symptom checklists.

## Mechanism
Two broad strands are usually distinguished. *Theory-driven* approaches build a mechanistic model of a process (for example value learning, perceptual inference, or evidence accumulation), fit it to behaviour or neuroimaging data, and ask which parameters differ between clinical groups or vary along symptom dimensions. *Data-driven* approaches use machine learning on large datasets to find predictive patterns, sometimes without a mechanistic model.

A typical theory-driven workflow is: (1) choose a task that isolates a computation, such as reversal learning or a probabilistic inference task; (2) specify candidate generative models of the participant's behaviour; (3) fit them and compare models by evidence (see [[model_selection]]); (4) interpret fitted parameters such as learning rates, noise, or the weight given to priors versus sensory evidence. Parameter recovery and model identifiability are persistent practical problems, and a parameter difference between groups does not by itself identify a cause.

Because many fitted models are descriptions of behaviour, claims about their neural or pharmacological implementation require separate evidence.

## Role in Active Inference
Active Inference supplies one prominent family of models in this field. In it, psychopathology is hypothesised to involve altered inference, most often framed as atypical *precision* (confidence) assigned to prior beliefs, sensory evidence, or action outcomes; see [[precision_weighting]] and [[prediction_error]]. The same framework covers perception, action, and interoception, which makes it attractive for conditions spanning several domains. These accounts are hypotheses under active research, with mixed empirical support and competing alternatives; they are not established causal explanations, and none of the models here is clinical guidance. The page [[psychiatry]] covers the free-energy-based formulations in more detail.

## Related concepts
- [[psychiatric_disorders]] — the clinical categories and dimensions that models try to explain
- [[schizophrenia]] — predictive-coding and aberrant-salience hypotheses
- [[autism]] — precision-based hypotheses about perception
- [[anxiety]] — hypotheses about uncertainty and threat estimation
- [[reinforcement_learning]] — a main formalism used in theory-driven work
- [[clinical_psychology]] — the applied discipline the field aims to inform

## References
- Montague, P. R., Dolan, R. J., Friston, K. J., & Dayan, P. (2012). Computational psychiatry. *Trends in Cognitive Sciences*.
- Friston, K. J., Stephan, K. E., Montague, R., & Dolan, R. J. (2014). Computational psychiatry: the brain as a phantastic organ. *Lancet Psychiatry*.
- Huys, Q. J. M., Maia, T. V., & Frank, M. J. (2016). Computational psychiatry as a bridge from neuroscience to clinical applications. *Nature Neuroscience*.
