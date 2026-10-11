---
title: Directed Acyclic Graphs
description: Graphs of directed edges with no cycles, their terminology, topological ordering, use as Bayesian network structures and causal diagrams, and their role in the structure of Active Inference generative models.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - graph_theory
  - graphical_models
  - causality
semantic_relations:
  - type: relates
    links:
      - "[[graph_theory]]"
      - "[[bayesian_networks]]"
      - "[[d_separation]]"
      - "[[causal_inference]]"
      - "[[probabilistic_graphical_models]]"
---

# Directed Acyclic Graphs

## Overview
A directed acyclic graph (DAG) is a set of nodes joined by directed edges such that following edges can never return to the starting node. If there is an edge $X\to Y$, then $X$ is a parent of $Y$ and $Y$ a child of $X$; ancestors and descendants extend these relations along directed paths. Every DAG admits a topological ordering in which each node comes after all its parents, and the absence of cycles is exactly the condition for such an ordering to exist.

## Use in probabilistic models
A DAG over random variables $X_1,\dots,X_n$ defines a Bayesian network when the joint distribution factorises as
$$
p(x_1,\dots,x_n)=\prod_{i=1}^{n}p\big(x_i\mid \mathrm{pa}(x_i)\big),
$$
where $\mathrm{pa}(x_i)$ are the parents of $x_i$. The graph encodes conditional independencies, which can be read off by d-separation. Different DAGs can encode the same set of independencies; such graphs form a Markov equivalence class, so structure usually cannot be recovered uniquely from observational data alone. The number of possible DAGs grows super-exponentially with the number of nodes, which makes exhaustive structure search infeasible beyond small problems.

## Use in causal modelling
When edges are read as direct causal influence, the DAG is a causal diagram. Together with functional equations for each variable, it forms a structural causal model, in which an intervention on a variable amounts to deleting its incoming edges and fixing its value (Pearl, 2009). The acyclicity assumption excludes feedback within a single time slice; feedback is represented by unrolling variables over time, giving a DAG over time-indexed copies.

## Relation to Active Inference
The relation is close to definitional. The discrete-time generative model of Active Inference, with hidden states, observations, policies and parameters, is a DAG, typically drawn as a Bayesian network or factor graph, and inference proceeds by message passing on that structure. Unrolling over time converts temporal dependence into acyclic structure. Where a model contains reciprocal coupling, such as between an agent and its environment, the cycle is resolved by indexing variables by time rather than by relaxing acyclicity.

## Related concepts
- [[graph_theory]] — the general theory of nodes and edges
- [[bayesian_networks]] — probabilistic models defined on DAGs
- [[d_separation]] — graphical criterion for conditional independence
- [[do_calculus]] — rules for interventions in causal DAGs
- [[factor_graphs]] — alternative representation for message passing

## References
- Pearl, J. (2009). Causality: Models, Reasoning, and Inference (2nd ed.). Cambridge University Press.
- Koller, D., & Friedman, N. (2009). Probabilistic Graphical Models: Principles and Techniques. MIT Press.
- Lauritzen, S. L. (1996). Graphical Models. Oxford University Press.
