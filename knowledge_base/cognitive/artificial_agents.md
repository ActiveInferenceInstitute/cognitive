---
title: Artificial Agents
type: knowledge_base
status: stable
created: 2026-02-07
updated: 2026-10-09
tags:
  - cognitive
  - active_inference
semantic_relations:
  - type: relates_to
    links:
      - "[[knowledge_base/cognitive/active_inference]]"
      - "[[knowledge_base/agents/README]]"
      - "[[knowledge_base/agents/architectures_overview]]"
      - "[[knowledge_base/cognitive/epistemic_foraging]]"
---

# Artificial Agents

## Overview

An artificial agent built on Active Inference is a program that carries a generative model of its environment, uses it to infer hidden states from observations, and chooses actions by comparing the predicted consequences of candidate policies. It has no separate reward function. Preferences enter as prior beliefs about which observations the agent expects to encounter, and one family of quantities, free energy, scores both perception and planning. The discrete-state formulation is well established in the literature (Friston et al. 2017; Da Costa et al. 2020; Parr, Pezzulo and Friston 2022). Scaling it to large state spaces, learned models and deep hierarchies is an active research programme, not a settled engineering recipe.

## Design Principles

**A generative model.** In the discrete case the model is a partially observable Markov decision process with a likelihood $A$, transitions $B$ conditioned on action, log-preferences $C$ over observations, a prior $D$ over initial states and a prior $E$ over actions. Designing an agent is mostly the work of choosing state, observation and action spaces that make these matrices small enough to specify and rich enough to matter.

**Belief updating.** Given an observation the agent combines its predicted state distribution with the likelihood of that observation, by exact Bayes, a variational approximation, a mean-field factorisation or sampling. See [[knowledge_base/cognitive/belief_updating]].

**Policy selection.** Each candidate action sequence is scored by its expected free energy, which in the canonical form is a risk term, the divergence between predicted and preferred observations, plus an ambiguity term, the expected entropy of the likelihood. Algebraically, risk plus ambiguity equals expected preference violation minus the expected information gain about hidden states, so the agent is drawn to policies that reveal state. This is the sense in which exploration is epistemic foraging ([[knowledge_base/cognitive/epistemic_foraging]]) and needs no added exploration bonus. See [[knowledge_base/mathematics/expected_free_energy]].

**Precision.** A temperature or precision parameter controls how deterministically the agent follows the best-scoring policy and is itself updated from prediction error.

## Practical Limits

Exhaustive policy enumeration grows as the number of actions raised to the planning horizon, so any implementation needs a cap or a pruning rule. Preferences must be hand-specified or learned. Whether the framework offers advantages over reinforcement learning on benchmark tasks is debated; the clearest differences are principled exploration and an explicit uncertainty-aware model, not raw performance.

## In This Repository

The package under `code/tools/src/models/active_inference` implements the discrete case:

- `ActiveInferenceModel` is the abstract base class. It defines belief updating, policy inference, precision updating, a free-energy readout and save/load of a `ModelState` holding beliefs, policies, precision, free energy and prediction error.
- `DiscreteGenerativeModel` is a validated, immutable container for $A$, $B$, $C$, $D$ and $E$, with methods for state prediction, posterior computation and expected free energy returned as total, risk, ambiguity and an epistemic diagnostic.
- `ActiveInferenceDispatcher` routes belief and policy inference to `variational`, `mean_field` or `sampling` methods. Only discrete policies are implemented, and first actions can be scored by their best continuation (`min`) or by a marginal over policies (`marginal`).
- `HomeostaticInference` specialises the base class with `HomeostaticControl` and `AdaptiveControl` modes.

Worked agents live under `code/Things`: `Simple_POMDP` (educational discrete agent), `Continuous_Generic` (`ContinuousActiveInference`, gradient descent on free energy over generalized coordinates) and `Generic_Thing`. The `Ant_Colony` and `KG_Multi_Agent` entries describe multi-agent designs; see [[knowledge_base/cognitive/multi_agent_active_inference]] and [[knowledge_base/cognitive/swarm_intelligence]]. The `cognitive-benchmark` command times the dispatcher, Simple POMDP and continuous agent on fixed inputs.

## References

- Friston, K., FitzGerald, T., Rigoli, F., Schwartenbeck, P., Pezzulo, G. (2017). Active inference: a process theory. *Neural Computation*.
- Da Costa, L., Parr, T., Sajid, N., Veselic, S., Neacsu, V., Friston, K. (2020). Active inference on discrete state-spaces: a synthesis. *Journal of Mathematical Psychology*.
- Parr, T., Pezzulo, G., Friston, K. (2022). *Active Inference: The Free Energy Principle in Mind, Brain, and Behavior*. MIT Press.

## Related Resources

- [[knowledge_base/cognitive/active_inference]]
- [[knowledge_base/agents/README]]
