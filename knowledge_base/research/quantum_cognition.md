---
title: Quantum Cognition
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
      - "[[knowledge_base/cognitive/active_inference]]"
      - "[[knowledge_base/research/quantum_systems]]"
      - "[[knowledge_base/mathematics/probability_theory]]"
      - "[[knowledge_base/mathematics/bayes_theorem]]"
      - "[[knowledge_base/cognitive/decision_making]]"
wikidata: Q7269019
---

# Quantum Cognition

## Overview

Quantum cognition uses the mathematics of quantum probability, not quantum physics in the brain, to model human judgement and decision-making. The claim is about the structure of the probability calculus: it asks whether the Kolmogorov axioms of classical probability are the right framework for describing how people reason under uncertainty. It should be kept distinct from proposals that quantum effects in neurons produce consciousness, such as the Penrose-Hameroff orchestrated objective reduction idea, which is a separate and strongly contested physical hypothesis (Tegmark's 2000 decoherence estimate is a standard objection).

## The formalism

A belief state is a unit vector $|\psi\rangle$ in a complex Hilbert space. A question or event corresponds to a subspace with orthogonal projector $P$, and the probability of a "yes" answer is $\|P|\psi\rangle\|^2$ (the Born rule). Answering a question updates the state by projecting onto the corresponding subspace and renormalising. If two projectors $P_A$ and $P_B$ do not commute, then the probability of "A then B" differs from "B then A", which gives a principled account of order effects. In classical probability, events are subsets of a sample space and the order of conjunction never matters.

For the same reason, the law of total probability can fail. A classical model requires $p(B) = p(A)p(B|A) + p(\neg A)p(B|\neg A)$. When the state is a superposition with respect to $A$, an interference term appears, which has been used to model the disjunction effect (Tversky and Shafir, 1992) and some violations of the sure-thing principle.

## Phenomena addressed

Quantum models have been applied to the conjunction fallacy, question-order effects in surveys (Wang and Busemeyer's 2013 order-effect model made a parameter-free prediction that was tested on large survey data), similarity and categorisation judgments, and contextuality in perception. Busemeyer and Bruza's *Quantum Models of Cognition and Decision* (2012) is the standard text.

## Status

This is an active and debated research area. Supporters point to compact models that fit order effects and make a priori predictions. Critics note that classical hierarchical Bayesian and heuristic models can also account for many of these effects, that the number of free parameters and the choice of dimension matter for model comparison, and that Hilbert-space structure may be a convenient description rather than evidence of an underlying mechanism.

## Relation to Active Inference

Standard Active Inference is built on classical probability, so quantum cognition is neither part of it nor required by it. Three connections are worth noting. First, both are normative-descriptive frameworks that treat judgement as inference over structured beliefs, so quantum-like models could be nested inside a generative-model description by modifying the belief space. Second, a separate literature extends the free energy principle to quantum systems, notably the work of Fields, Friston, Glazebrook and Levin (2022), which concerns physical systems and not human judgement; see [[knowledge_base/research/quantum_systems]]. Third, order effects could in principle be modelled classically through context-dependent priors or precision, which would make the comparison between the two accounts an empirical question. No result establishes that Active Inference needs quantum probability.

## In this repository

See [[knowledge_base/mathematics/probability_theory]] and [[knowledge_base/mathematics/bayes_theorem]] for the classical baseline, [[knowledge_base/cognitive/decision_making]] for the phenomena, and [[active_inference_quantum_cognitive_systems_path]] for a study path.

## References

- Busemeyer, J. R., & Bruza, P. D. (2012). *Quantum Models of Cognition and Decision*. Cambridge University Press.
- Pothos, E. M., & Busemeyer, J. R. (2013). Can quantum probability provide a new direction for cognitive modeling? *Behavioral and Brain Sciences*, 36, 255-274.
- Fields, C., Friston, K., Glazebrook, J. F., & Levin, M. (2022). A free energy principle for generic quantum systems. *Progress in Biophysics and Molecular Biology*, 173, 36-59.

## Related Resources

- [[knowledge_base/cognitive/active_inference]]
- [[knowledge_base/research/quantum_systems]]
