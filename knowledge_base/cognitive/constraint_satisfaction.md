---
title: Constraint Satisfaction
description: Cognitive processing modelled as the simultaneous weighing of many soft constraints to find the best-fitting interpretation, with applications to sentence processing and a link to energy minimisation.
type: concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - language
  - connectionism
  - inference
semantic_relations:
  - type: relates
    links:
      - "[[language_processing]]"
      - "[[reasoning_problem_solving]]"
      - "[[probabilistic_inference]]"
---

# Constraint Satisfaction

## Overview
Constraint satisfaction treats a cognitive task as the search for a state of a system that best satisfies many partially conflicting constraints at once. In cognitive modelling the constraints are usually soft, weighted preferences rather than inviolable rules, and processing is parallel. In the formal sense of artificial intelligence, a constraint satisfaction problem has variables, domains and hard constraints; the cognitive usage is looser.

## Formalism
In connectionist networks, units represent hypotheses and weights encode constraints: positive weights favour compatible hypotheses together, negative weights favour mutual exclusion. Activation spreads until the network settles into a pattern of high overall "goodness". Hopfield (1982) showed that symmetric networks descend an energy function, so settling minimises $E=-\tfrac12\sum_{ij}w_{ij}s_is_j$ and finds local optima; Rumelhart, Smolensky, McClelland and Hinton (1986) used goodness-of-fit maximisation to model schema and interpretation processes. Stochastic variants such as Boltzmann machines add noise to avoid poor local minima.

## Applications
- **Sentence processing**: constraint-based accounts hold that lexical frequency, verb argument preferences, semantic plausibility and context are integrated immediately in resolving ambiguity (MacDonald, Pearlmutter and Seidenberg, 1994). They contrast with two-stage models in which an initial syntactic analysis based on structural principles is revised if it fails (Frazier and Rayner, 1982), a debate that remains active (see [[language_processing]]).
- **Perception**: interpreting ambiguous or occluded input by satisfying multiple cues (see [[perceptual_organization]], [[context_effects]]).
- **Reasoning and problem solving**: finding assignments that satisfy a set of conditions (see [[reasoning_problem_solving]]).

## Role in Active Inference
Combining weighted constraints has a direct probabilistic reading. Each constraint contributes a factor to the joint probability, so the best interpretation is the maximum a posteriori configuration, and graded constraint strength corresponds to likelihood or prior precision (see [[probabilistic_inference]], [[precision_weighting]]). Hopfield-style energy minimisation and free-energy minimisation are closely related: the free energy of a stochastic network equals energy minus entropy, and variational message passing on factor graphs implements multi-constraint integration (see [[factor_graphs]], [[message_passing]]). The correspondence is exact for specific model classes, but treating all cognitive constraint satisfaction as Active Inference goes beyond the evidence; most psycholinguistic models do not use the framework.

## Related concepts
- [[parallel_processing]] — constraints are evaluated simultaneously
- [[recurrent_networks]] — settling dynamics
- [[free_energy_minimization]] — the objective with an energy-based analogue

## References
- Hopfield, J. J. (1982). Neural networks and physical systems with emergent collective computational abilities. PNAS, 79, 2554-2558.
- Frazier, L., & Rayner, K. (1982). Making and correcting errors during sentence comprehension: Eye movements in the analysis of structurally ambiguous sentences. Cognitive Psychology, 14, 178-210.
- Rumelhart, D. E., Smolensky, P., McClelland, J. L., & Hinton, G. E. (1986). Schemata and sequential thought processes in PDP models. In Parallel Distributed Processing, Vol. 2. MIT Press.
- MacDonald, M. C., Pearlmutter, N. J., & Seidenberg, M. S. (1994). The lexical nature of syntactic ambiguity resolution. Psychological Review, 101, 676-703.
