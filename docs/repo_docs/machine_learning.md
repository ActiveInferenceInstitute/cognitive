---
title: Machine learning and learning in this package
type: guide
status: stable
created: 2024-02-12
updated: 2026-10-10
tags:
  - machine-learning
  - learning
  - guide
semantic_relations:
  - type: implements
    links:
      - "[[docs/repo_docs/ai_validation_framework|Validation framework]]"
  - type: relates
    links:
      - "[[knowledge_base/cognitive/active_inference|Active Inference]]"
      - "[[knowledge_base/cognitive/predictive_processing|Predictive processing]]"
---

# Machine learning and learning in this package

The package is not a machine-learning library. It has no training loops,
datasets, optimizers, or dependency on PyTorch, TensorFlow, JAX, or scikit-learn;
its runtime dependencies are `click`, `jinja2`, `matplotlib`, `networkx`,
`numpy`, `Pillow`, `plotly`, `PyYAML`, and `seaborn`. "Learning" here means the
following, all in NumPy.

## What learns

| Mechanism | Where | What changes |
|---|---|---|
| Belief updating | `ActiveInferenceDispatcher.dispatch_belief_update` | State beliefs move toward the posterior; `InferenceConfig.learning_rate` in (0, 1] sets the step size |
| Precision updating | `ActiveInferenceModel.update_precision(beta)`, `ActiveInferenceDispatcher.update_precision` | Precision tracks the inverse of recent prediction error |
| Continuous precision learning | `ContinuousActiveInference(precision_learning_rate=...)` | Observation precision adapts when the rate is positive (default `0.0`, so off) |
| Adaptive exploration | `HomeostaticInference.update_parameters({"stability_index": ...})` | For an adaptive controller, `exploration_weight` is set from the stability index and clipped to 0.1 to 0.5 |

The generative model matrices (`A`, `B`, `C`, `D`, `E`) are fixed once a
`DiscreteGenerativeModel` is built; the package does not learn them from data.
`InferenceConfig` takes a `learning_rate`, a `temperature`, and a
`num_samples` for the sampling method, none of which trains parameters.

## If you want to learn parameters

Fit the arrays outside the package (for example by counting transitions in your
own data), then pass them to `DiscreteGenerativeModel`, which validates shapes
and normalization. Keep that fitting code in your own project or add it here
with tests in `code/tests/` and an entry in `CHANGELOG.md`.

## Background reading in the knowledge base

- [[knowledge_base/cognitive/active_inference|Active Inference]] and
  [[knowledge_base/cognitive/predictive_processing|Predictive processing]]
- [[knowledge_base/cognitive/reinforcement_learning|Reinforcement learning]],
  [[knowledge_base/cognitive/deep_learning|Deep learning]], and
  [[knowledge_base/cognitive/policy_learning|Policy learning]]
- [[knowledge_base/mathematics/parameter_learning|Parameter learning]]

These pages are conceptual. They describe the ideas, not code in the package.

## Related documentation

- [[docs/api/api_reference|API reference]]
- [[docs/repo_docs/model_implementation|Model implementation]]
- [[docs/repo_docs/validation|Validation]]
