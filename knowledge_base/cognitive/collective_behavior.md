---

title: Collective Behavior

type: knowledge_base

status: stable

tags: [collective, swarm, coordination, multi-agent]

semantic_relations:

  - type: extends

    links: [swarm_intelligence, stigmergic_coordination]

  - type: relates

    links: [social_insect_cognition, cooperation, active_inference]

---

# Collective Behavior

Collective behavior arises when local interactions among many agents produce coherent global patterns (alignment, flocking, task allocation). In an active inference view, each agent minimizes its expected free energy given beliefs that include expectations about others, yielding coordination through shared priors, signals, and environmental traces.

## Mechanisms

- Local rules: attraction/repulsion/alignment; message passing via observations

- Stigmergy: environment-mediated coupling (see [[stigmergic_coordination]])

- Role differentiation: implicit task allocation under resource and uncertainty constraints

## Minimal model

```math

q_i(\pi) = \operatorname{softmax}(-\gamma_i G_i(\pi;\, o_i, m_i))\,,\quad

m_i \equiv \text{beliefs about neighbors/environment}

```

Coordination emerges when priors and likelihoods couple across agents through observations and shared artifacts.

## Applications

- Distributed sensing and coverage

- Foraging and transport in social insects

- Multi-robot task allocation and navigation

See also: [[social_insect_cognition]], [[swarm_intelligence]], [[cooperation]].

