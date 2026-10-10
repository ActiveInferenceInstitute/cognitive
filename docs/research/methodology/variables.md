---
title: Variables
type: research
status: stable
updated: 2026-10-10
semantic_relations:
  - type: relates
    links:
      - "[[docs/api/api_reference]]"
      - "[[docs/config/README]]"
---

# Variables

The inputs of an experiment are the model matrices (`A, B, C, D, E`) and the
`InferenceConfig` fields: `method`, `policy_type`, `temporal_horizon`,
`learning_rate`, `precision_init`, and the optional `num_samples`, `temperature`,
`discount_factor`, `exploration_weight`, `policy_limit`, `policy_aggregation`, and
`seed`. Out-of-range values raise `ValueError`. Record all of them with a result.
`exploration_weight` defaults to `0.0`; a positive value adds an extra
information-seeking bias (see `CHANGELOG.md`). The full list and the YAML shape of
the agent configurations are in [[docs/api/api_reference|the API reference]]; the
repository's own configuration files are described in [[docs/config/README|the
configuration docs]].
