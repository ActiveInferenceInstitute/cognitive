---
title: Prior Beliefs
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - active_inference
  - bayesian_inference
  - generative_model
aliases:
  - prior_knowledge
semantic_relations:
  - type: relates
    links:
      - "[[active_inference]]"
      - "[[generative_model]]"
      - "[[bayesian_inference]]"
---

# Prior Beliefs

## Overview
A prior belief is a probability distribution held before conditioning on the current observation. In Active Inference every part of the generative model that is not an observation is a prior of some kind: over initial states, over outcomes the agent expects or prefers, over actions, and over the model's own parameters.

## Core claims
In the discrete formulation implemented in `DiscreteGenerativeModel` (`code/tools/src/models/active_inference/generative_model.py`), the priors are explicit arrays.

| Object | Convention in this repo | Role |
|---|---|---|
| `D[s]` | prior over initial hidden states | where the agent believes it starts |
| `C[o]` | log preferences over observations | prior preferences; `preferences` is softmax(C) |
| `E[a]` | prior over actions | habit; enters the policy posterior as ln E |
| `A[o, s]`, `B[s', s, a]` | column-stochastic likelihood and transitions | beliefs about how the world generates outcomes |

`A` and `B` are priors in the broader sense of fixed beliefs about causal structure; the validator rejects any whose columns do not sum to one.

**Priors as preferences.** Because C is a distribution over outcomes, the risk term in expected free energy is KL[q(o|π) || p*(o)] with p*(o) = softmax(C). Wanting an outcome and expecting it are the same object, which is how the framework answers the objection in [[dark_room_problem]].

**Empirical priors.** In hierarchical models, the posterior at one level supplies the prior for the level below; priors are then learned from data rather than stipulated. **Parameter priors** are Dirichlet distributions over `A`, `B` and `D`, and a precision prior governs how confident policy selection is.

**Prior enters inference twice.** Belief updating computes posterior ∝ prior × likelihood (the `posterior` method multiplies the prior over states by the row `A[o]` and normalizes). Policy inference adds the action prior: the dispatcher's mean-field method uses logits ln E − G/T, and the variational and sampling methods subtract ln(prior) from G only when a `goal_prior` is passed, which avoids counting the prior twice.

## Role in Active Inference
Priors are where the agent's goals, habits and assumptions live, and they are chosen by the modeller. The framework does not say where preferences originally come from; the origin of C is an open issue (see [[preference_learning]]). Misspecified priors produce misbehaviour that is easy to misread as a fault of inference.

## Related concepts
- [[generative_model]] — the structure that contains the priors
- [[belief_initialization]] — choosing D
- [[belief_updating]] — how a prior becomes a posterior
- [[preference_learning]] — learning C
- [[hierarchical_inference]] — empirical priors across levels
- [[knowledge_base/mathematics/expected_free_energy]] — where C enters policy scoring
- [[belief_systems]] — priors as entrenched beliefs
