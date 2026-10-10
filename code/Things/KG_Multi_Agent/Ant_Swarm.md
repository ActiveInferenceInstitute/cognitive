---
title: Ant Swarm design note
type: concept
status: draft
updated: 2026-10-10
semantic_relations:
  - type: related_to
    links:
      - "[[code/Things/KG_Multi_Agent/README|KG Multi-Agent overview]]"
      - "[[code/Things/Ant_Colony/README|Ant Colony]]"
---

# Ant Swarm design note

Design sketch; no code in this repository.

## Setting

Active Inference POMDP agents ("ants") traverse an Obsidian knowledge graph.
The graph is held as a dict $\mathcal{G}$ in which each node stores its
contents (string), a semantic embedding of those contents (array), and its
neighbours (names of nodes it links to).

## Trial

1. The user gives a prompt. It is used directly or rewritten as a hypothesis
   $h_0$ that makes a definitive claim using key ontological terms; its
   embedding is $e_0$.
2. Similarity scores between $e_0$ and node embeddings (or, for a hierarchical
   graph, nodes in the top layer) select the top $k$ nodes. One agent starts at
   each of them and carries that node's contents, embedding, and neighbours.
3. At each time step an agent moves along edges from its start node, using its
   current embedding as the reference. Every gathered node is added to the
   agent's local graph $G_k$, which has the same structure as $\mathcal{G}$ and
   so records its path.
4. The agent summarizes the collected nodes into a new node whose contents are
   an Obsidian-style summary with `[[links]]` to every node visited. The node
   gets a unique hypothesis ID in its name, is added to $\mathcal{G}$, and its
   embedding becomes the agent's new reference embedding.

Agents could carry a POMDP or hybrid generative model for navigation: an
observation modality saying whether the current node was already visited, a
policy horizon for planning a path, preferences over similarity scores, and
actions about when and how to write summary nodes. Clone-structured cognitive
graphs could serve as maps of the explored graph.

## Relation to ants and stigmergy

- Decentralised decisions: like ants following local pheromone cues, each
  agent decides from local similarity scores and its own local graph.
- Indirect communication: summary nodes left in $\mathcal{G}$ play the role of
  pheromone deposits, since other agents can discover and build on them.
- Iterative refinement: each summary becomes the next hypothesis, as reinforced
  trails guide later foraging.
- Emergence: the graph grows through many agents' summaries, without a central
  controller.

These are analogies for design, not results from any experiment in this
repository.
