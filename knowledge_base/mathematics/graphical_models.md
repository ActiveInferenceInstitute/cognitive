---
title: Graphical Models
type: mathematical_concept
status: stable
created: 2024-03-20
tags:
  - mathematics
  - graphical_models
  - probability
  - inference
  - machine_learning
semantic_relations:
  - type: foundation_for
    links:
      - [[factor_graphs]]
      - [[belief_propagation]]
      - [[message_passing]]
  - type: includes
    links:
      - [[bayesian_networks]]
      - [[markov_random_fields]]
      - [[conditional_independence]]
  - type: related
    links:
      - [[probabilistic_models]]
      - [[variational_inference]]
      - [[active_inference_theory]]
---

# Graphical Models

## Overview

Probabilistic graphical models (PGMs) are a framework for representing complex probability distributions using graph-based structures. They combine graph theory and probability theory to provide a compact, interpretable representation of joint distributions over many variables.

Graphical models are foundational to [[active_inference_theory|active inference]] and [[free_energy_principle|free energy principle]] approaches, providing the mathematical substrate for [[generative_models|generative models]] of the environment.

## Types of Graphical Models

### Directed Models (Bayesian Networks)

[[bayesian_networks|Bayesian networks]] use directed acyclic graphs (DAGs) to represent conditional dependencies:

$$p(x_1, \ldots, x_n) = \prod_{i=1}^{n} p(x_i | \text{pa}(x_i))$$

where $\text{pa}(x_i)$ denotes the parent nodes of $x_i$.

**Properties:**
- Encode causal or generative structure
- Natural representation for [[generative_models|generative models]]
- Used in [[active_inference_pomdp|active inference POMDP]] formulations

### Undirected Models (Markov Random Fields)

[[markov_random_fields|Markov random fields]] use undirected graphs with potential functions:

$$p(x) = \frac{1}{Z} \prod_{c \in \mathcal{C}} \psi_c(x_c)$$

where $\mathcal{C}$ denotes the set of cliques and $Z$ is the partition function.

**Properties:**
- Encode symmetric relationships
- Natural for spatial/relational data
- [[conditional_independence|Conditional independence]] via graph separation

### Factor Graphs

[[factor_graphs|Factor graphs]] provide a unified representation for both directed and undirected models:

$$p(x) = \frac{1}{Z} \prod_{a} f_a(x_a)$$

**Properties:**
- Unify directed and undirected representations
- Enable efficient [[message_passing|message passing]] algorithms
- Basis for [[belief_propagation|belief propagation]]

## Inference in Graphical Models

### Exact Inference

- **Variable elimination**: Marginalize variables in sequence
- **Junction tree algorithm**: Transform to tree structure for efficient computation
- **Belief propagation on trees**: Exact marginals via message passing

### Approximate Inference

- **Loopy belief propagation**: [[belief_propagation|BP]] on graphs with cycles
- **Variational inference**: [[variational_methods|Variational methods]] minimize KL divergence
- **Monte Carlo methods**: Sampling-based approximations
- **Expectation propagation**: Moment-matching approximations

### Learning

- **Parameter learning**: Maximum likelihood, Bayesian estimation
- **Structure learning**: Discovering graph structure from data
- **EM algorithm**: Learning with latent variables

## Connection to Active Inference

### Generative Models

In [[active_inference_theory|active inference]], the agent's internal model is a graphical model specifying:

- **Hidden states** ($s$): Unobserved environmental variables
- **Observations** ($o$): Sensory data
- **Actions** ($a$): Agent's control variables
- **Parameters** ($\theta$): Model parameters

### Hierarchical Models

[[hierarchical_processing|Hierarchical]] graphical models capture multi-scale structure:

$$p(o, s^{(1)}, \ldots, s^{(L)}) = p(o | s^{(1)}) \prod_{l=1}^{L-1} p(s^{(l)} | s^{(l+1)}) p(s^{(L)})$$

### Dynamic Models

Temporal graphical models (e.g., Hidden Markov Models, Dynamic Bayesian Networks) represent processes unfolding over time, central to [[active_inference_pomdp|POMDP]] formulations.

## Key Concepts

### Conditional Independence

[[conditional_independence|Conditional independence]] is encoded by graph structure:

- **d-separation** (directed graphs)
- **Graph separation** (undirected graphs)
- **Markov blanket**: Minimal set rendering a node independent of all others

### Information Theory

- **Entropy** of distributions
- **Mutual information** between variables
- **KL divergence** for model comparison

### Exponential Family

Many graphical models use exponential family distributions:

$$p(x | \eta) = h(x) \exp(\eta^T T(x) - A(\eta))$$

enabling efficient natural gradient methods and [[variational_methods|variational inference]].

## Applications

- [[bayesian_brain|Bayesian Brain]] hypothesis
- [[predictive_processing|Predictive coding]] in neural circuits
- [[active_inference_pomdp|Decision making]] under uncertainty
- [[probabilistic_models|Probabilistic modeling]] in science
- Machine learning and AI

## Related Concepts

- [[factor_graphs]] - Bipartite graph representation
- [[belief_propagation]] - Message passing inference
- [[message_passing]] - General inference framework
- [[variational_inference]] - Approximate inference methods
- [[probabilistic_models]] - Probabilistic modeling
- [[markov_random_fields]] - Undirected graphical models
- [[conditional_independence]] - Independence structure
- [[probabilistic_graphical_models]] - Extended treatment

## References

- Koller, D. & Friedman, N. (2009). Probabilistic Graphical Models: Principles and Techniques
- Bishop, C. M. (2006). Pattern Recognition and Machine Learning, Ch. 8
- Jordan, M. I. (2004). Graphical Models
- Wainwright, M. J. & Jordan, M. I. (2008). Graphical Models, Exponential Families, and Variational Inference
