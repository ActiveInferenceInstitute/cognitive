---
title: Difference Reduction
description: A problem-solving strategy in which the solver compares the current state with the goal, selects an operator that reduces the difference, and sets up subgoals when the operator cannot yet be applied.
type: concept
status: stable
created: 2026-10-12
updated: 2026-10-12
tags:
  - problem_solving
  - means_ends_analysis
  - heuristics
  - cognitive_psychology
semantic_relations:
  - type: relates
    links:
      - "[[reasoning_problem_solving]]"
      - "[[heuristics]]"
      - "[[insight_problem_solving]]"
      - "[[working_memory]]"
      - "[[goal_setting]]"
      - "[[protocol_analysis]]"
---

# Difference Reduction

## Overview
Difference reduction is a general-purpose method for problem solving in which the solver repeatedly notes how the current state differs from the goal state and chooses an action that diminishes that difference. When the preferred action cannot be taken because its preconditions are not met, the solver sets the removal of that obstacle as a new subgoal, and the cycle repeats. The full procedure is called means-ends analysis. The method was central to the theory of human problem solving proposed by Allen Newell, Cliff Shaw and Herbert Simon in the late 1950s and was implemented in their General Problem Solver (GPS), and it is described in detail in Newell and Simon's *Human Problem Solving* (1972).

## The procedure
In outline, a means-ends solver does the following.
1. Compare the current state $s$ with the goal $g$ and identify a difference $d$.
2. Select an operator associated with reducing differences of type $d$.
3. If the operator is applicable in $s$, apply it to obtain a new state and go to step 1.
4. Otherwise create a subgoal of transforming $s$ into a state where the operator can be applied, and solve that subproblem in the same way.

The method uses a table connecting kinds of difference to the operators that tend to remove them, and it does not require a complete search of the problem space. It therefore keeps the demands on short-term memory low, which fits the limited capacity of human working memory (see [[working_memory]]).

## Evidence and use
Newell and Simon built their account from think-aloud protocols of people solving problems such as logic derivations and puzzles, and compared the sequences of goals and subgoals in those protocols with the behaviour of GPS (see [[protocol_analysis]]). The approach also influenced later work on planning in artificial intelligence.

## Limitations
Because each step is chosen to decrease the apparent distance from the goal, the method is a form of hill climbing and can be trapped when the solution requires a move that temporarily increases the difference. Problems in which the path to the goal begins by moving away from it, or in which the representation hides the true structure of the difference, are therefore harder for solvers who rely on this strategy (see [[insight_problem_solving]]).

## Relation to Active Inference
Difference reduction and Active Inference are different frameworks, and the relation between them is an analogy. A preference over observations, such as a goal state, has a role in Active Inference similar to the goal in means-ends analysis, and the selection of policies that are expected to bring observations toward the preferred ones resembles reduction of a difference (see [[policy_selection]]). However, policies in Active Inference are scored by expected free energy, which includes an epistemic term that values information gain, so an agent can choose to move away from the goal in order to resolve uncertainty. In that respect the free energy approach differs from a pure hill-climbing strategy.

## References
- Newell, A., Shaw, J. C., & Simon, H. A. (1958). Elements of a theory of human problem solving. Psychological Review, 65(3), 151-166. doi:10.1037/h0048495
- Newell, A., & Simon, H. A. (1972). Human Problem Solving. Prentice-Hall.
- Ericsson, K. A., & Simon, H. A. (1984). Protocol Analysis: Verbal Reports as Data. MIT Press.
