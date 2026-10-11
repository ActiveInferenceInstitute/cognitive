---
title: Dynamic Programming
description: Solving sequential optimisation problems by recursively decomposing them into subproblems via the Bellman principle, with value and policy iteration, and its relation to planning in Active Inference.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - optimal_control
  - planning
  - bellman_equation
semantic_relations:
  - type: relates
    links:
      - "[[optimal_control]]"
      - "[[markov_decision_process]]"
      - "[[value_function]]"
      - "[[hamilton_jacobi_bellman]]"
      - "[[policy_selection]]"
---

# Dynamic Programming

## Overview
Dynamic programming (DP) is a method for solving optimisation problems whose solutions have optimal substructure: an optimal plan for the whole problem contains optimal plans for its remaining subproblems. Bellman (1957) formalised this as the principle of optimality and used it to treat multistage decision problems. Storing solutions to subproblems avoids recomputing them, and this reuse is what separates DP from naive enumeration of all action sequences, whose number grows exponentially with the horizon.

## The Bellman equation
For a Markov decision process with reward $r(s,a)$, transition probabilities $P(s'\mid s,a)$ and discount $\gamma\in[0,1)$, the optimal value function satisfies
$$
V^{*}(s)=\max_{a}\Big[r(s,a)+\gamma\sum_{s'}P(s'\mid s,a)\,V^{*}(s')\Big].
$$
Main algorithms:
- **Backward induction**: for a finite horizon, compute the value at the final step and work backwards, recording the best action at each stage.
- **Value iteration**: repeatedly apply the Bellman operator, which is a contraction for $\gamma<1$ and so converges to $V^{*}$.
- **Policy iteration**: alternate between evaluating a fixed policy and improving it greedily; it terminates in finitely many steps for finite problems.

The main limitation is the curse of dimensionality: the table of values grows exponentially with the number of state variables, which motivates function approximation and approximate DP. In continuous time the recursion becomes the Hamilton-Jacobi-Bellman partial differential equation.

## Relation to Active Inference
Planning in discrete-time Active Inference has traditionally enumerated whole policies and scored each with expected free energy, which scales as in naive search. Recursive formulations, such as sophisticated inference (Friston et al., 2021), instead evaluate the expected free energy of actions by a backward-style tree search conditioned on future beliefs, in the spirit of DP. The correspondence is structural: expected free energy replaces reward in the objective, and the recursion is over beliefs rather than states. Whether such schemes recover the Bellman-optimal policy depends on how preferences and epistemic terms are specified, and it is an object of current analysis.

## Related concepts
- [[markov_decision_process]] — the standard setting for the Bellman equation
- [[value_function]] — the object that DP computes
- [[optimal_control]] — the continuous-time control counterpart
- [[hamilton_jacobi_bellman]] — the continuous-time Bellman equation
- [[reinforcement_learning]] — learns the same quantities without a known model
- [[belief_space_planning]] — DP applied to belief states

## References
- Bellman, R. (1957). Dynamic Programming. Princeton University Press.
- Bertsekas, D. P. (2017). Dynamic Programming and Optimal Control, Vol. I (4th ed.). Athena Scientific.
- Friston, K., Da Costa, L., Hafner, D., Hesp, C., & Parr, T. (2021). Sophisticated inference. Neural Computation, 33, 713-763.
