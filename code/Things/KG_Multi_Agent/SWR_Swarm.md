---
title: Sharp-wave-ripple swarm design note
type: concept
status: draft
updated: 2026-10-10
semantic_relations:
  - type: related_to
    links:
      - "[[code/Things/KG_Multi_Agent/README|KG Multi-Agent overview]]"
      - "[[code/Things/KG_Multi_Agent/Ant_Swarm|Ant Swarm]]"
---

# Sharp-wave-ripple swarm design note

Design sketch; no code in this repository. The snippets are illustrative
pseudocode, not an API of this repository, and none of them is run or tested.

The note extends [[code/Things/KG_Multi_Agent/Ant_Swarm|Ant Swarm]] with ideas
taken from the hippocampus literature on sharp wave-ripples (SPW-Rs), replay,
and memory consolidation (associated with György Buzsáki's work). The mapping is
an analogy for design, and no claim here is a result.

## Design directives

1. **Graph as memory network.** Each node of the global graph $\mathcal{G}$ is a
   memory trace with contents, an embedding, a timestamp (episodic context), and
   weighted edges whose strength reflects how often replay reinforced them.
2. **Agents as replay mechanisms.** Agents traverse the graph to strengthen
   associations. Awake replay revisits recently traversed paths during pauses in
   exploration. Compressed replay re-traverses a stored sequence quickly, in the
   manner of SPW-R reactivation.
3. **Temporal dynamics.** Timestamps feed a relevance score that multiplies
   embedding similarity by an exponential decay in the age of the node, used to
   prioritise which neighbour to visit.
4. **Consolidation.** During replay an agent synthesizes a node that summarizes
   its traversed path and links to the contributing nodes, then adds it to the
   global graph.
5. **Navigation.** Preconfigured trajectories, analogous to place-cell
   sequences, guide movement and are updated from replay and POMDP policies.
6. **Synchronization.** Local graphs are merged into the global graph during
   replay, so synthesized nodes become visible to other agents.
7. **Triggers.** Awake replay fires during exploration pauses; compressed replay
   fires after significant events or at regular intervals.
8. **Forgetting.** Edge strengths decay with time unless replay or synthesis
   reinforces them.

## Pseudocode

```text
node  = {contents, embedding, neighbors: {name: strength}, timestamp}

relevance(node)  = exp(-age(node) / decay_constant) * similarity(agent.embedding, node.embedding)

awake_replay(agent):
    for each edge (u, v) inside agent.local_graph:
        global_graph[u].neighbors[v] += small_increment

compressed_replay(agent):
    for consecutive nodes (u, v) in agent.trajectory:
        global_graph[u].neighbors[v] += larger_increment

synthesize(agent):
    summary = summarize(contents of agent.local_graph)
    add node(summary, embed(summary), links to agent.local_graph) to global_graph

navigate(agent):
    return argmax over neighbors n of relevance(n)

decay(graph):
    for each edge: strength *= exp(-age / decay_constant)
```

## Repository support

Nothing above is implemented. The increments, decay constant, and trigger rules
are free parameters that a real implementation would have to choose and test.
