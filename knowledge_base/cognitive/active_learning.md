---
title: Active Learning
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - learning
  - exploration
  - information_seeking
semantic_relations:
  - type: relates
    links:
      - "[[active_inference]]"
      - "[[epistemic_value]]"
      - "[[information_gain]]"
---

# Active Learning

## Overview
Active learning is learning in which the learner chooses which data to acquire. In machine learning the term specifically names methods that select which unlabeled examples to send to an oracle for labeling, in order to reach a given accuracy with fewer labels. In cognitive science the same idea appears as information-seeking, hypothesis-driven exploration, and curiosity-driven behavior. The two uses share the principle that data collection is itself a decision.

## Mechanism
Common query strategies in machine learning include:

- **Uncertainty sampling**: query the instance on which the current model is least certain (lowest confidence, highest entropy, or smallest margin).
- **Query by committee**: query where an ensemble of models disagrees most.
- **Expected model change / expected error reduction**: query the instance expected to change the model, or reduce risk, most.
- **Expected information gain**: choose the query that maximizes the expected reduction in entropy of the parameter posterior, $\mathbb{E}_{y}\left[\mathrm{KL}\big(p(\theta \mid x, y)\,\|\,p(\theta)\big)\right]$.

Pitfalls include sampling bias, outliers that look maximally uncertain but are uninformative, and the cost of recomputing the model for each query. Experiments in children and adults show that people often choose queries that approximately maximize expected information, though with systematic deviations.

## Role in Active Inference
Active inference gives a normative principle for exactly this behavior. Policies are scored by [[expected_free_energy]], which splits into a pragmatic term (expected preference satisfaction) and an epistemic term (expected information gain about hidden states). When the model is also learning its parameters, an additional novelty term rewards actions expected to reduce uncertainty about those parameters, so the agent is drawn to observations that improve its model (Schwartenbeck et al., 2019). The result is that exploration and exploitation come from one objective, rather than from an added exploration bonus or a separate query heuristic, and the balance changes automatically as uncertainty is resolved. Two cautions apply. The correspondence to machine-learning active learning is formal, not a specific algorithm: the agent acts in an environment rather than querying a labeling oracle. And the expected-free-energy decomposition rests on assumptions about the generative model that are currently debated in the literature, so the formulation is an actively revised theory.

## Related concepts
- [[epistemic_value]] — the information-seeking component of policy value
- [[information_gain]] — quantity maximized by informative queries
- [[exploration_exploitation]] — trade-off that active learning addresses
- [[expected_free_energy]] — objective that scores queries and actions
- [[action_selection]] — where the query decision is made
- [[unsupervised_learning]] — the setting in which labeled queries are most valuable to avoid

## References
- Settles, B. (2009). *Active Learning Literature Survey*. University of Wisconsin-Madison Computer Sciences Technical Report 1648.
- Schwartenbeck, P., Passecker, J., Hauser, T. U., FitzGerald, T. H. B., Kronbichler, M., & Friston, K. J. (2019). Computational mechanisms of curiosity and goal-directed exploration. *eLife*.
