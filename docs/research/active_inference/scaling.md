---
title: Scaling limits of the discrete agent
type: research
status: stable
updated: 2026-10-10
semantic_relations:
  - type: relates
    links:
      - "[[docs/api/api_reference]]"
      - "[[knowledge_base/mathematics/policy_selection]]"
---

# Scaling limits of the discrete agent

The dispatcher plans by enumerating every action sequence up to
`temporal_horizon`, so the number of policies is `num_actions ** horizon`.
`InferenceConfig.policy_limit` (default 4096) caps the enumeration; when the cap
is too low to cover at least one policy per first action, the dispatcher raises a
`ValueError` that names the limit instead of silently planning with a subset.

`DiscreteGenerativeModel.enumerate_policies(horizon, limit=None)` exposes the
same enumeration.

`cognitive-benchmark` times a 64 by 64 softmax, one dispatcher belief update, one
`SimplePOMDP` step, and one continuous step, and prints JSON (`--output` also
writes it to a file). It does not time policy enumeration, so the repository
makes no measured claim about how planning cost grows. Hierarchical or sampled
planning is not built ([[hierarchical]]).
