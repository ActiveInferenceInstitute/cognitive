---
title: Replicator Dynamics
description: Differential equations describing how the frequencies of competing strategies or types change in proportion to relative fitness, their game-theoretic stability results and a reading as Bayesian updating.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - evolutionary_dynamics
  - game_theory
  - population_dynamics
semantic_relations:
  - type: relates
    links:
      - "[[evolutionary_dynamics]]"
      - "[[evolutionary_game_theory]]"
      - "[[lotka_volterra]]"
      - "[[probability_simplex]]"
      - "[[natural_selection]]"
      - "[[population_dynamics]]"
---

# Replicator Dynamics

## Overview
The replicator equation describes the evolution of the relative frequencies of $n$ types in a population. Let $x_i$ be the frequency of type $i$, so that $x$ lies on the probability simplex, and let $f_i(x)$ be its fitness. Then
$$\dot{x}_i=x_i\big(f_i(x)-\bar f(x)\big),\qquad \bar f(x)=\sum_j x_jf_j(x).$$
Types fitter than the population average increase in frequency and others decrease. The simplex is invariant, because the right-hand sides sum to zero. In evolutionary game theory the fitness is the expected payoff, often linear, $f(x)=Ax$ for a payoff matrix $A$.

## Stability results
Taylor and Jonker introduced the equation for evolutionarily stable strategies. Standard results, often grouped under the folk theorem of evolutionary game theory, state that every Nash equilibrium is a rest point, every stable rest point is a Nash equilibrium, and every evolutionarily stable strategy is an asymptotically stable rest point. A strict Nash equilibrium is asymptotically stable. The converses can fail. Dynamics may cycle, as in the rock-paper-scissors game, and may display more complex behaviour in larger systems. A transformation of variables shows that the replicator equation on $n$ types is equivalent to a Lotka-Volterra system with $n-1$ species.

## Interpretations
The equation is the continuous-time form of selection in a large population with reproduction proportional to fitness. It also describes social learning by imitation under suitable assumptions, so the same equations appear in models of cultural change.

## Relation to Active Inference
The link is formal and suggestive. In discrete time, the update $x_i'\propto x_if_i$ has the same form as Bayes theorem with prior $x_i$ and likelihood $f_i$, so selection can be read as inference in which frequencies are beliefs and fitness plays the role of evidence. Work on free energy and evolution develops this analogy, treating adaptive dynamics as a form of Bayesian model selection by the environment. The correspondence holds for the stated equations and need not carry over to richer models of evolution, which include mutation and finite populations.

## Related concepts
- [[evolutionary_dynamics]] — the broader setting of selection and drift
- [[evolutionary_game_theory]] — the game-theoretic source of the model
- [[lotka_volterra]] — an equivalent system after a change of variables
- [[probability_simplex]] — the state space of type frequencies
- [[natural_selection]] — the process the equation idealises

## References
- Taylor, P. D., & Jonker, L. B. (1978). Evolutionary stable strategies and game dynamics. Mathematical Biosciences, 40, 145-156.
- Hofbauer, J., & Sigmund, K. (1998). Evolutionary Games and Population Dynamics. Cambridge University Press.
- Nowak, M. A. (2006). Evolutionary Dynamics: Exploring the Equations of Life. Harvard University Press.
