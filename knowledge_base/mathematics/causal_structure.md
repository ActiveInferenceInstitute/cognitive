---
title: Causal Structure
description: The pattern of cause-effect dependencies among variables, usually encoded as a directed acyclic graph, with the Markov condition, equivalence classes, and the role of structure in Active Inference generative models.
type: mathematical_concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - causality
  - graphical_models
  - generative_models
semantic_relations:
  - type: relates
    links:
      - "[[causal_inference]]"
      - "[[bayesian_networks]]"
      - "[[generative_models]]"
      - "[[conditional_independence]]"
---

# Causal Structure

## Overview
The causal structure of a system is the specification of which variables directly influence which others. It is normally drawn as a directed acyclic graph (DAG) with an arrow $X\to Y$ when $X$ is a direct cause of $Y$ relative to the variables in the model. In the sense used on this page it is a modelling object, distinct from the "cause-effect structure" of integrated information theory, which is defined over the physical mechanisms of a system.

## Formalism
- **Parents and factorisation.** The causal Markov condition says every variable is independent of its non-descendants given its parents, which gives $p(x_1,\dots,x_n)=\prod_i p(x_i\mid \mathrm{pa}_i)$.
- **Interventions as graph surgery.** Setting $X=x$ removes the arrows into $X$. The remaining factors stay invariant, which is the modularity assumption that distinguishes causal from merely statistical structure ([[causal_inference]]).
- **Observational equivalence.** Different DAGs can encode the same conditional independences. Chains $A\to B\to C$, $A\leftarrow B\leftarrow C$ and the fork $A\leftarrow B\to C$ all imply $A\perp C\mid B$ and are Markov equivalent, but the collider $A\to B\leftarrow C$ implies marginal independence of $A$ and $C$ and is distinguishable. Observational data therefore identify structure only up to an equivalence class (CPDAG) unless interventions, time order or functional assumptions are added.
- **Faithfulness** assumes that all independences in the data come from the graph, and not from cancelling pathways. It is an assumption, and can fail.
- **Cycles and feedback.** Static DAGs do not represent feedback. Dynamical systems unroll it over time, so that $x_t$ causes $x_{t+1}$, or use structural models with cycles.

## Role in Active Inference
The generative model of an Active Inference agent has an explicit factorisation: hidden states cause observations through $A$, evolve through $B$ under actions, and are initialised by $D$. This factorisation is the model's assumed causal structure, and the agent's inferences are only as good as the match between that structure and the world's. Structure learning in this setting selects among candidate factorisations by comparing free-energy bounds on evidence; because evidence alone cannot separate Markov-equivalent graphs, action and intervention supply the extra information. The factor graph of a model is a computational structure for message passing and does not need to coincide with causal structure, so one should not read causal claims off a factor graph without checking.

## Related concepts
- [[causal_inference]] — estimation of intervention effects on a given structure
- [[bayesian_networks]] — DAG-based probabilistic models
- [[d_separation]] — reads independences off the graph
- [[generative_models]] — the models whose factorisation embodies assumed structure
- [[factor_graphs]] — computational graph that may differ from the causal one
- [[structure_learning]] — inferring structure from data
