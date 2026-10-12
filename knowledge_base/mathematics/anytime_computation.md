---
title: Anytime Computation
description: Algorithms that can be interrupted at any moment and return a result whose quality improves with run time, with their formal properties, examples, and relevance to bounded inference and planning.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - bounded_rationality
  - algorithms
  - planning
semantic_relations:
  - type: relates
    links:
      - "[[active_inference_loop]]"
      - "[[computational_complexity]]"
      - "[[variational_inference]]"
      - "[[markov_chain_monte_carlo]]"
      - "[[policy_selection]]"
---

# Anytime Computation

## Overview
An anytime algorithm can be stopped at any time and will return a valid answer, with the quality of the answer improving as more computation is allowed. The concept originated in work on planning under time pressure by Dean and Boddy and was developed by Horvitz and by Zilberstein to formalise the trade-off between decision quality and the cost of delay.

## Properties
Zilberstein (1996) lists desirable properties of anytime algorithms:
- **Measurable quality**: the quality of a result can be assessed, or at least estimated.
- **Monotonicity**: quality does not decrease with additional time.
- **Consistency**: quality is a predictable function of time and input.
- **Diminishing returns**: improvement per unit time shrinks as the run lengthens.
- **Interruptibility**: the algorithm can be stopped at an arbitrary time, whereas a *contract* algorithm must be told its time allocation in advance and may produce no useful output if stopped earlier.

The behaviour of an algorithm is summarised by a performance profile, giving expected quality as a function of run time. Profiles allow a meta-level controller to decide when to stop by comparing the marginal gain in quality with the cost of time, which is the basis of rational metareasoning. A contract algorithm can be converted into an interruptible one, at a bounded cost in time, by running it repeatedly with increasing contracts.

Examples include iterative deepening search, local search, gradient-based and message-passing inference, Monte Carlo sampling and Monte Carlo tree search. Algorithms that return an answer only at termination, such as many direct solvers, lack the property.

## Relation to Active Inference
Belief updating in Active Inference is typically an iterative descent on variational free energy, so it has an anytime character: each iteration produces a better approximation to the posterior, and in a real-time agent the process can be cut off by the arrival of the next observation, with the best current beliefs used for action (see [[active_inference_loop]]). This relation is structural and is not a formal result for any specific implementation. Whether the properties above, such as monotone improvement, hold depends on the scheme, and the same consideration applies to policy evaluation over large policy spaces, where planning must be truncated (see [[policy_selection]]).

## Related concepts
- [[active_inference_loop]] — the perception-action cycle in which time limits arise
- [[computational_complexity]] — the cost that anytime methods trade against quality
- [[variational_inference]] — iterative approximation with anytime behaviour
- [[markov_chain_monte_carlo]] — sampling whose accuracy grows with run length
- [[policy_selection]] — planning that may need truncation
- [[free_energy_minimization]] — the iterative process that is interruptible

## References
- Dean, T., & Boddy, M. (1988). An analysis of time-dependent planning. Proceedings of the Seventh National Conference on Artificial Intelligence (AAAI-88), 49-54.
- Zilberstein, S. (1996). Using anytime algorithms in intelligent systems. AI Magazine, 17, 73-83.
- Russell, S., & Norvig, P. (2021). Artificial intelligence: A modern approach (4th ed.). Pearson.
