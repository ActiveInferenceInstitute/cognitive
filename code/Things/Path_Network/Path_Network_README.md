---
title: Path Network design notes
type: concept
status: draft
updated: 2026-10-10
semantic_relations:
  - type: related_to
    links:
      - "[[code/Things/Path_Network/README|Path Network overview]]"
      - "[[code/Things/Continuous_Generic/Continuous_Generic_README|Continuous Generic agent]]"
---

# Path Network design notes

Design sketch; no code in this repository. This folder holds documentation
only. There is no `path_network` package, no `config/`, `core/`, `utils/` or
`simulation/` directory, and nothing here can be imported or run.

## Idea

Simulate a randomly generated topology of nodes whose edges encode effective
causal influence, where every node is a continuous-time active inference agent.
Nodes could differ in their integration time step and in the order of the
generalized-coordinate (Taylor series) expansion they use for anticipatory
inference.

The environment is pictured as nested sine waves forming the "level of water
in the world". Each node is a ship or tower that infers how far to ascend or
descend so that it stays within a tolerable range of the sea level, which
floats every boat.

## Building blocks

- Per-node inference: the repository's only continuous-time agent is
  `ContinuousActiveInference` in `code/Things/Continuous_Generic`. It performs
  precision-weighted gradient descent over generalized coordinates with a
  configurable time step and number of orders, which is the closest existing
  analogue of a node.
- Topology and environment: a graph library and a sum of sinusoids would be
  needed. Neither exists in this folder.

## Open design questions

- How influence weights between nodes enter each node's generative model.
- Whether the topology adapts during a run, and on what signal.
- What the action is (height change) and how it feeds back on the water level.
