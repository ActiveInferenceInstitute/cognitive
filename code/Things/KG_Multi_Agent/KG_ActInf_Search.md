---
title: Knowledge graph search with an active inference agent
type: concept
status: draft
updated: 2026-10-10
semantic_relations:
  - type: related_to
    links:
      - "[[code/Things/KG_Multi_Agent/README|KG Multi-Agent overview]]"
      - "[[code/Things/Simple_POMDP/Simple_POMDP_README|SimplePOMDP]]"
---

# Knowledge graph search with an active inference agent

Design sketch; no code in this repository. The pseudocode below is not an API
of the `cognitive` package and does not run.

## Specification

- A prompt goes to an LLM. Its system prompt names the key terms the LLM must
  return to denote which hidden state the prompt relates to. The returned terms
  are the discrete observations of the POMDP agent.
- The agent infers one hidden-state factor. Each hidden state is a node of an
  Obsidian `[[linked]]` knowledge graph.
- The number of actions equals the number of hidden states: an action means
  visiting a node.
- Policy inference over a scalar horizon `policy_horizon` selects a sequence of
  `[[links]]` to visit.
- The markdown of the nodes in the selected policy is concatenated in order and
  passed to the LLM as graph-RAG material for answering the original query.

## Pseudocode

```text
nodes     = names of notes in the vault          # hidden states = actions
A         = P(key-term observation | node)       # from the LLM's key-term vocabulary
B[a]      = transition when visiting node a
C, D      = preferences over observations, prior over nodes
agent     = discrete POMDP agent(A, B, C, D)

function answer(query):
    reply        = llm(system_prompt, query)
    observation  = key_terms(reply)
    beliefs      = agent.update_beliefs(observation)
    policy       = agent.infer_policy(beliefs, horizon = policy_horizon)
    context      = join(read_note(node) for node in policy)
    return llm(query, context)
```

## Repository support

`Things.Simple_POMDP.SimplePOMDP` plans one step ahead, so a multi-step policy
over nodes would use `ActiveInferenceDispatcher` and `DiscreteGenerativeModel`
from the `cognitive` package; see [[docs/api/api_reference|the API reference]].
The LLM client, key-term extraction, and note reader are not provided.
