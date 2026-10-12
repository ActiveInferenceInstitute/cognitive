---
title: Intervention Calculus
description: The formal treatment of interventions in causal models, including the do-operator, graph surgery, truncated factorisation and Pearl's rules for converting interventional into observational probabilities.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - causal_inference
  - interventions
  - graphical_models
semantic_relations:
  - type: relates
    links:
      - "[[do_calculus]]"
      - "[[causal_inference]]"
      - "[[causal_structure]]"
      - "[[directed_acyclic_graphs]]"
      - "[[d_separation]]"
---

# Intervention Calculus

## Overview
Observing that a variable $X$ takes the value $x$ is different from setting $X$ to $x$ by an external action. Intervention calculus gives the second operation a formal meaning inside a structural causal model. The notation $P(y\mid do(x))$ denotes the distribution of $Y$ when $X$ is forced to $x$, as opposed to $P(y\mid x)$, which describes the population in which $X$ happened to equal $x$.

## Definition and structure
In a causal model over a directed acyclic graph with parents $\mathrm{pa}_i$, the joint distribution factorises as $P(v)=\prod_i P(v_i\mid \mathrm{pa}_i)$. The intervention $do(X=x)$ replaces the mechanism for $X$ by a constant, which corresponds to deleting the arrows into $X$ (graph surgery). The result is the truncated factorisation
$$P(v\mid do(x))=\prod_{i:\,V_i\ne X}P(v_i\mid \mathrm{pa}_i)\quad\text{for } v \text{ consistent with } x.$$
If a set $Z$ satisfies the back-door criterion relative to $(X,Y)$, then $P(y\mid do(x))=\sum_z P(y\mid x,z)P(z)$, the adjustment formula. This differs from conditioning, which would weight $z$ by $P(z\mid x)$.

Pearl's calculus consists of three rules, each stating when a term can be inserted or deleted, or an action exchanged for an observation, given a graphical independence condition evaluated in a modified graph. Together they are complete for identifying causal effects from observational data and graph structure, as shown independently by Shpitser and Pearl and by Huang and Valtorta in 2006. When no sequence of rule applications removes all $do$ operators, the effect is not identifiable from the available data.

## Role in Active Inference
An acting agent changes its environment by intervening, and in discrete-state Active Inference actions are selected by comparing the expected outcomes of alternative policies, a computation that has the same form as evaluating $P(\text{outcome}\mid do(\text{policy}))$ under the agent's generative model. Whether the agent's model is genuinely causal, rather than merely predictive, depends on whether its structure reflects the true mechanisms. Work on structure learning in Active Inference and on causal inference by active agents explores this connection, and it is an active research topic rather than a settled result.

## Related concepts
- [[do_calculus]] — the three inference rules in detail
- [[causal_inference]] — the wider field of estimating causal effects
- [[causal_structure]] — the graph encoding the mechanisms
- [[directed_acyclic_graphs]] — the structure on which truncation is defined
- [[d_separation]] — the graphical criterion used in the rules
- [[expected_free_energy]] — scoring of policies by anticipated outcomes

## References
- Pearl, J. (1995). Causal diagrams for empirical research. Biometrika, 82, 669-688.
- Pearl, J. (2009). Causality: Models, Reasoning, and Inference (2nd ed.). Cambridge University Press.
- Spirtes, P., Glymour, C., & Scheines, R. (2000). Causation, Prediction, and Search (2nd ed.). MIT Press.
