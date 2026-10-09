---
title: Belief Systems
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - epistemology
  - philosophy
  - belief
semantic_relations:
  - type: relates
    links:
      - "[[active_inference]]"
      - "[[knowledge_base/cognitive/belief_updating]]"
      - "[[epistemology]]"
---

# Belief Systems

## Overview
A belief system is a structured set of interdependent beliefs, together with the relations that make some beliefs support, presuppose or constrain others. The philosophical questions concern how such systems are justified, how they change under evidence, and why parts of them resist revision.

## Core claims
**Web of belief.** Quine and Ullian's image treats beliefs as a network: beliefs at the periphery meet experience directly and are revised readily, while central ones (logic, core physical principles) are revised only at great cost to everything else. Evidence therefore confronts the system holistically, which is the Duhem-Quine thesis.

**Coherentism and foundationalism.** Coherentists justify beliefs by mutual support within the system; foundationalists require basic beliefs that need no further support. See [[epistemology]] for the free-energy treatment of this contrast.

**Belief revision.** The AGM framework (Alchourrón, Gärdenfors, Makinson) states rationality postulates for expansion, contraction and revision of a belief set, with a notion of entrenchment ordering what to give up first. Bayesian accounts replace sets with credences and revision with conditionalization.

**Resistance and perseverance.** Empirically, people retain beliefs after the evidence for them is discredited. Whether this reflects irrationality or rational entrenchment of strongly weighted central priors is contested.

## Role in Active Inference
"Belief" has a technical sense here: a belief is a probability distribution q(s) over hidden states, not a propositional attitude, and its update is posterior inference (see [[knowledge_base/cognitive/belief_updating]]). A belief system maps onto a hierarchical generative model, where higher levels hold slowly changing priors that act as empirical priors for lower levels. Entrenchment then corresponds to prior precision: high-precision priors move little under prediction error (see [[precision_weighting]]). This gives a mechanistic reading of the web-of-belief picture, though it is an interpretation, not a derivation from Quine.

Computational psychiatry uses the same machinery to describe delusion, for example as aberrant precision on priors or on prediction errors (Adams and colleagues); these accounts are hypotheses under test, not settled results. The repository's `ActiveInferenceModel` holds a single flat belief vector, so the hierarchical structure above is described in [[hierarchical_inference]] rather than implemented here.

## Related concepts
- [[belief_updating]] — how beliefs change given an observation
- [[belief_evolution]] — beliefs over longer timescales
- [[hierarchical_inference]] — levels of beliefs and empirical priors
- [[precision_weighting]] — the mechanism for entrenchment
- [[generative_model]] — the structure that organizes beliefs
- [[reasoning_problem_solving]] — reasoning from and about belief

## References
- Quine, W. V. O. & Ullian, J. S. (1970). *The Web of Belief*.
- Alchourrón, C., Gärdenfors, P. & Makinson, D. (1985). On the logic of theory change. *Journal of Symbolic Logic*.
- Adams, R. A. et al. (2013). The computational anatomy of psychosis. *Frontiers in Psychiatry*.
