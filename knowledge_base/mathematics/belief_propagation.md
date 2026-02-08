---
title: Belief Propagation
type: mathematical_concept
status: stable
created: 2024-03-20
tags:
  - mathematics
  - inference
  - graphical_models
  - message_passing
  - algorithms
semantic_relations:
  - type: implements
    links:
      - [[message_passing]]
      - [[factor_graphs]]
      - [[variational_inference]]
  - type: foundation_for
    links:
      - [[active_inference_theory]]
      - [[free_energy_principle]]
      - [[belief_updating]]
  - type: related
    links:
      - [[graphical_models]]
      - [[markov_random_fields]]
      - [[probabilistic_models]]
---

# Belief Propagation

## Overview

Belief propagation (BP), also known as sum-product message passing, is an algorithm for performing inference in graphical models. It computes marginal distributions for unobserved nodes conditioned on observed nodes by passing local messages between nodes in a factor graph.

Belief propagation is foundational to [[active_inference_theory|active inference]] and the [[free_energy_principle|free energy principle]], providing the computational mechanism for [[belief_updating|belief updating]] in hierarchical generative models.

## Mathematical Formulation

### Message Passing on Factor Graphs

In a [[factor_graphs|factor graph]], messages are passed between variable nodes and factor nodes:

**Variable-to-factor message:**

$$\mu_{x \to f}(x) = \prod_{g \in \text{ne}(x) \setminus f} \mu_{g \to x}(x)$$

**Factor-to-variable message:**

$$\mu_{f \to x}(x) = \sum_{\sim x} f(X) \prod_{y \in \text{ne}(f) \setminus x} \mu_{y \to f}(y)$$

where $\text{ne}(\cdot)$ denotes the neighboring nodes and $\sum_{\sim x}$ denotes summation over all variables except $x$.

### Marginal Computation

The marginal (belief) at variable node $x$ is computed as the product of all incoming messages:

$$b(x) \propto \prod_{f \in \text{ne}(x)} \mu_{f \to x}(x)$$

### Convergence Properties

- **Trees**: BP computes exact marginals in a single forward-backward pass
- **Loopy graphs**: BP becomes approximate (loopy belief propagation) but often provides good approximations
- **Convergence guarantees**: Guaranteed on trees; on loopy graphs, convergence depends on graph structure and potential strength

## Variants

### Sum-Product Algorithm

The standard BP algorithm computes marginal probabilities:

$$p(x_i | \text{evidence}) = \frac{1}{Z} \prod_{f \in \text{ne}(x_i)} \mu_{f \to x_i}(x_i)$$

### Max-Product Algorithm

Computes the most probable configuration (MAP inference):

$$\mu_{f \to x}(x) = \max_{\sim x} f(X) \prod_{y \in \text{ne}(f) \setminus x} \mu_{y \to f}(y)$$

### Expectation Propagation

An extension that uses moment matching to handle non-conjugate models, projecting messages onto exponential family distributions.

### Variational Message Passing

Connects BP to [[variational_methods|variational inference]] by interpreting messages as natural parameters of variational distributions.

## Connection to Active Inference

### Hierarchical Predictive Coding

In [[predictive_processing|predictive processing]], belief propagation manifests as:

- **Bottom-up messages**: Prediction errors from sensory data
- **Top-down messages**: Predictions from higher-level beliefs
- **Lateral messages**: Precision-weighted integration

### Free Energy Minimization

BP can be understood as performing [[free_energy_minimization|free energy minimization]]:

$$F = \sum_i D_{KL}[q(x_i) \| p(x_i | \text{pa}(x_i))] - \sum_i \mathbb{E}_{q}[\ln p(o_i | x_i)]$$

where the message passing updates correspond to coordinate descent on the variational free energy.

### Generative Models

In [[active_inference_theory|active inference]], belief propagation operates on the agent's [[generative_models|generative model]] to:

1. **Infer hidden states** from observations
2. **Predict future outcomes** under different policies
3. **Evaluate expected free energy** for action selection

## Implementation Considerations

### Computational Complexity

- **Trees**: $O(n \cdot k^w)$ where $n$ is nodes, $k$ is state cardinality, $w$ is tree width
- **Loopy graphs**: Per-iteration cost same as trees, but may require multiple iterations
- **Gaussian models**: Efficient closed-form message computation

### Numerical Stability

- Use log-domain computation to avoid underflow
- Normalize messages at each step
- Apply damping for loopy BP convergence

### Software Implementations

- **RxInfer.jl**: Reactive message passing framework
- **pymdp**: Python implementation for discrete active inference
- **SPM**: MATLAB toolbox for neuroimaging with BP-based inference

## Applications in Cognitive Modeling

- [[bayesian_brain|Bayesian Brain]] hypothesis
- [[predictive_processing|Predictive processing]] in cortical hierarchies
- [[active_inference_pomdp|Active inference POMDP]] for decision making
- Perceptual inference
- [[cognitive/attention_mechanisms|Attention]] as precision optimization

## Related Concepts

- [[message_passing]] - General framework for message-based inference
- [[factor_graphs]] - Graphical representation for factorized distributions
- [[variational_inference]] - Variational approach to approximate inference
- [[belief_updating]] - Bayesian updating of beliefs
- [[graphical_models]] - Probabilistic graphical model frameworks
- [[markov_random_fields]] - Undirected graphical models
- [[free_energy_principle]] - Unifying principle connecting BP to cognition

## References

- Pearl, J. (1988). Probabilistic Reasoning in Intelligent Systems
- Kschischang, F. R., Frey, B. J., & Loeliger, H.-A. (2001). Factor graphs and the sum-product algorithm
- Yedidia, J. S., Freeman, W. T., & Weiss, Y. (2003). Understanding belief propagation and its generalizations
- Friston, K. J. et al. (2017). Active inference and belief propagation
- Parr, T. & Friston, K. J. (2019). Generalised free energy and active inference
