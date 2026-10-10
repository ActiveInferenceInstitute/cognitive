---
title: Active inference applications in this repository
type: research
status: stable
updated: 2026-10-10
semantic_relations:
  - type: relates
    links:
      - "[[docs/api/agent_api]]"
      - "[[docs/guides/application/README]]"
      - "[[knowledge_base/research/README]]"
---

# Active inference applications in this repository

Three kinds of agent can be run today; each is described in
[[docs/api/agent_api|the agent API page]] and [[docs/api/api_reference|the API reference]].

- `Things.Simple_POMDP.SimplePOMDP`: a small configurable discrete agent.
- `Things.Continuous_Generic.ContinuousActiveInference`: a continuous-state agent
  in generalized coordinates with a visualizer.
- `HomeostaticInference` (from `HomeostaticFactory`): discrete control toward a
  target state.

The application areas discussed elsewhere (robotics, ecology, economics, security,
and others) are documentation only. Their theory notes are in
`knowledge_base/research/`, and the learning paths are in
[[docs/guides/application/README|the application guides]]. A note that describes an
application does not imply that a model of it exists in `code/`.
