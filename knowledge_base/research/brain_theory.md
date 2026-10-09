---
title: Brain Theory
type: knowledge_base
status: stable
created: 2026-02-07
updated: 2026-10-09
tags:
  - research
  - active_inference
semantic_relations:
  - type: relates_to
    links:
      - "[[knowledge_base/cognitive/bayesian_brain_hypothesis]]"
      - "[[knowledge_base/cognitive/predictive_processing]]"
      - "[[knowledge_base/cognitive/bayesian_brain]]"
      - "[[knowledge_base/cognitive/predictive_coding]]"
      - "[[knowledge_base/cognitive/neural_coding]]"
      - "[[knowledge_base/cognitive/hierarchical_processing]]"
      - "[[knowledge_base/philosophy/dark_room_problem]]"
      - "[[knowledge_base/research/clinical_neuroscience]]"
      - "[[knowledge_base/biology/neuroscience]]"
---

# Brain Theory

## Overview

Brain theory seeks general principles that explain what neural tissue computes and how. Two ideas have organised much of it in recent decades: that the brain performs a form of Bayesian inference about the causes of its sensory input, and that cortex implements this through hierarchical prediction and prediction-error signalling. Both have older roots. Helmholtz described perception as unconscious inference in the nineteenth century, and the modern formulation was developed by Knill and Pouget (2004) as the "Bayesian brain" programme.

## Main questions

- **Bayesian perception.** Behavioural studies show that people often combine cues and priors in ways close to Bayes-optimal, for example in cue integration and in sensorimotor estimation.
- **Neural coding.** How populations of neurons might represent probability distributions: probabilistic population codes, sampling-based codes, and parametric codes.
- **Cortical hierarchies.** Rao and Ballard (1999) showed that a hierarchy in which higher areas predict lower-level activity, and lower areas send up the residual error, can account for some response properties in visual cortex. Friston (2005) generalised this to a theory of cortical responses, and Bastos and colleagues (2012) proposed how the computations could map onto the canonical cortical microcircuit.
- **Learning.** How the structure of the generative model and its parameters are acquired.

## Connection to Active Inference

Active Inference extends this line from perception to action. Whereas the Bayesian brain account concerns inference about hidden causes, Active Inference adds that actions are selected to fulfil predictions or to reduce uncertainty, so that behaviour is a part of the same inferential scheme.

The empirical status differs by layer. The claim that perception is well approximated by Bayesian inference, within limits, has substantial behavioural support. Evidence for predictive coding in specific neural circuits is mixed: some findings, such as repetition suppression and mismatch responses, fit predictions, while others have alternative explanations, and the proposed laminar assignment of prediction and error units is under test. The strongest claim, that the free energy principle explains brain function in general, is a theoretical position that has been criticised as too flexible to be falsified. Jones and Love's (2011) critique of "Bayesian fundamentalism" raised related concerns about Bayesian models in cognitive science. A notable puzzle, addressed in the repository's [[knowledge_base/philosophy/dark_room_problem]] page, is why an agent that minimises surprise does not simply seek a predictable environment.

## In this repository

The core accounts are [[knowledge_base/cognitive/bayesian_brain]], [[knowledge_base/cognitive/predictive_coding]], [[knowledge_base/cognitive/neural_coding]] and [[knowledge_base/cognitive/hierarchical_processing]]. Clinical consequences are in [[knowledge_base/research/clinical_neuroscience]], and the biological side in [[knowledge_base/biology/neuroscience]]. A study path is [[active_inference_neuroscience_learning_path]].

## References

- Knill, D. C., & Pouget, A. (2004). The Bayesian brain: the role of uncertainty in neural coding and computation. *Trends in Neurosciences*, 27, 712-719.
- Rao, R. P. N., & Ballard, D. H. (1999). Predictive coding in the visual cortex. *Nature Neuroscience*, 2, 79-87.
- Friston, K. (2005). A theory of cortical responses. *Philosophical Transactions of the Royal Society B*, 360, 815-836.
- Bastos, A. M., et al. (2012). Canonical microcircuits for predictive coding. *Neuron*, 76, 695-711.
- Jones, M., & Love, B. C. (2011). Bayesian fundamentalism or enlightenment? *Behavioral and Brain Sciences*, 34, 169-188.

## Related Resources

- [[knowledge_base/cognitive/bayesian_brain_hypothesis]]
- [[knowledge_base/cognitive/predictive_processing]]
