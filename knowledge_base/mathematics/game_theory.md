---
title: Game Theory
type: mathematical_concept
status: stable
created: 2024-03-20
tags:
  - mathematics
  - game_theory
  - decision_making
  - strategy
  - multi_agent
semantic_relations:
  - type: foundation_for
    links:
      - [[knowledge_base/cognitive/decision_making]]
      - [[knowledge_base/cognitive/social_cognition]]
      - [[knowledge_base/cognitive/cooperation]]
  - type: related
    links:
      - [[optimization_theory]]
      - [[probability_theory]]
      - [[information_theory]]
  - type: applied_in
    links:
      - [[code/Things/Baseball_Game/AGENTS]]
      - [[knowledge_base/cognitive/collective_behavior]]
---

# Game Theory

## Overview

Game theory is the mathematical study of strategic interaction among rational decision-makers. It provides formal frameworks for analyzing situations where the outcome for each participant depends on the actions of all participants.

In the context of [[knowledge_base/cognitive/active_inference|active inference]] and [[free_energy_principle|free energy principle]], game theory connects to multi-agent inference problems where agents must model each other's beliefs and policies.

## Core Concepts

### Normal-Form Games

A game in normal form is defined by:

- **Players**: $N = \{1, 2, \ldots, n\}$
- **Strategy sets**: $S_i$ for each player $i$
- **Payoff functions**: $u_i: S_1 \times \cdots \times S_n \to \mathbb{R}$

### Nash Equilibrium

A strategy profile $s^* = (s_1^*, \ldots, s_n^*)$ is a Nash equilibrium if:

$$u_i(s_i^*, s_{-i}^*) \geq u_i(s_i, s_{-i}^*) \quad \forall s_i \in S_i, \forall i \in N$$

No player can improve their payoff by unilaterally changing strategy.

### Mixed Strategies

Players may randomize over pure strategies:

$$\sigma_i \in \Delta(S_i)$$

Nash's theorem guarantees existence of mixed strategy equilibria in finite games.

## Game Types

### Cooperative vs. Non-Cooperative

- **Non-cooperative**: Individual strategic decision-making
- **Cooperative**: Coalition formation and collective action ([[knowledge_base/cognitive/cooperation|cooperation]])

### Zero-Sum vs. General-Sum

- **Zero-sum**: One player's gain equals another's loss
- **General-sum**: Both cooperation and competition possible

### Static vs. Dynamic

- **Static (simultaneous)**: Players choose actions simultaneously
- **Dynamic (sequential)**: Players observe and respond to actions over time
- **Repeated games**: Interactions occur multiple times, enabling reciprocity

### Games of Incomplete Information

Bayesian games where players have private information:

$$p(\theta_i | \theta_{-i})$$

connecting to [[bayesian_inference|Bayesian inference]] and [[knowledge_base/cognitive/active_inference|active inference]].

## Connection to Active Inference

### Multi-Agent Active Inference

In multi-agent settings, each agent maintains a [[generative_models|generative model]] that includes:

- Models of other agents' beliefs and policies
- Expected responses to own actions
- [[knowledge_base/cognitive/social_cognition|Social cognition]] via theory of mind

### Expected Free Energy in Games

The expected free energy for agent $i$ incorporates:

$$G_i(\pi_i) = \mathbb{E}_{q(\tilde{o}, \tilde{s} | \pi_i, \pi_{-i})}[\ln q(\tilde{s}) - \ln p(\tilde{o}, \tilde{s})]$$

where $\pi_{-i}$ represents the inferred policies of other agents.

### Evolutionary Game Theory

Connects to [[knowledge_base/biology/evolutionary_biology|evolutionary biology]] through:

- **Replicator dynamics**: Population-level strategy evolution
- **Evolutionarily stable strategies (ESS)**: Robust equilibria
- **Fitness landscapes**: Strategy performance over time

## Applications

### Multi-Agent Systems

- [[knowledge_base/cognitive/collective_behavior|Collective behavior]] in agent societies
- [[knowledge_base/cognitive/social_cognition|Social cognition]] and opponent modeling
- Multi-agent coordination

### Biological Applications

- Predator-prey dynamics
- Mating strategies
- Resource competition
- [[knowledge_base/cognitive/cooperation|Cooperation]] evolution

### Cognitive Applications

- [[knowledge_base/cognitive/decision_making|Decision making]] under strategic uncertainty
- Theory of mind and recursive reasoning
- Communication and signaling games

## Related Concepts

- [[knowledge_base/cognitive/decision_making]] - Decision theory
- [[optimization_theory]] - Optimization methods
- [[probability_theory]] - Probabilistic foundations
- [[information_theory]] - Information measures
- [[knowledge_base/cognitive/social_cognition]] - Social reasoning
- [[knowledge_base/cognitive/cooperation]] - Cooperative behavior

## References

- von Neumann, J. & Morgenstern, O. (1944). Theory of Games and Economic Behavior
- Nash, J. (1950). Equilibrium Points in N-Person Games
- Maynard Smith, J. (1982). Evolution and the Theory of Games
- Yoshida, W. et al. (2008). Game Theory of Mind

## See also

- [[knowledge_base/research/concepts/game_theory|Game Theory]]
