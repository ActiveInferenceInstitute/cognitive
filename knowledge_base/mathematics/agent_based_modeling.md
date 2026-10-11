---
title: Agent-Based Modeling
description: A simulation approach in which many autonomous agents follow local rules and interact, so that population-level patterns emerge, with uses in ecology, social science and multi-agent Active Inference.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - simulation
  - emergence
  - complex_systems
semantic_relations:
  - type: relates
    links:
      - "[[complex_systems]]"
      - "[[emergence]]"
      - "[[swarm_intelligence]]"
      - "[[collective_behavior]]"
      - "[[multi_agent_active_inference]]"
---

# Agent-Based Modeling

## Overview
Agent-based modeling (ABM), also called individual-based modeling in ecology, represents a system as a collection of discrete agents, each with its own state and behavioural rules, situated in an environment. The model is run forward in time, and aggregate quantities are measured from the simulated population. The approach is generative: a macroscopic pattern is explained by exhibiting local rules that reproduce it. It is used where heterogeneity, spatial structure, adaptation or network interactions make closed-form aggregate equations hard to write down.

## Components and examples
A typical model specifies:
- **Agents**: internal state, perception of neighbours or local environment, and a decision rule, which may be fixed, stochastic or learned.
- **Environment**: a grid, continuous space or network that holds resources and mediates interaction.
- **Schedule**: the order or timing in which agents update, which can change results and should be reported.
- **Observables**: statistics collected across agents and runs, with repeated stochastic runs used to estimate variability.

Classic examples are Schelling's segregation model (1971), in which mild individual preferences about neighbours generate strong spatial segregation, the Boids flocking rules of Reynolds (1987), and the Sugarscape model of Epstein and Axtell (1996). Results depend on the rules chosen, so analysis normally includes sensitivity checks and comparison with data; a model that reproduces a pattern shows that the rules are sufficient, not that they are the actual cause.

## Relation to Active Inference
The link is an active modelling practice rather than a single theorem. Each agent in a simulation can be given a generative model and act by minimising expected free energy, replacing hand-written rules with inference and policy selection. Multi-agent studies of this kind examine communication, shared beliefs and collective behaviour, including models motivated by social insect colonies. ABM supplies the simulation substrate, while Active Inference supplies a principled, uniform specification of what each agent does.

## Related concepts
- [[complex_systems]] — the class of systems ABM is often used to study
- [[emergence]] — macroscopic patterns arising from local rules
- [[swarm_intelligence]] — collective problem solving by simple agents
- [[collective_behavior]] — group-level phenomena that ABM can reproduce
- [[multi_agent_active_inference]] — Active Inference agents interacting in a population

## References
- Schelling, T. C. (1971). Dynamic models of segregation. Journal of Mathematical Sociology, 1, 143-186.
- Reynolds, C. W. (1987). Flocks, herds and schools: A distributed behavioral model. ACM SIGGRAPH Computer Graphics, 21, 25-34.
- Epstein, J. M., & Axtell, R. (1996). Growing Artificial Societies: Social Science from the Bottom Up. Brookings Institution Press and MIT Press.
- Railsback, S. F., & Grimm, V. (2019). Agent-Based and Individual-Based Modeling: A Practical Introduction (2nd ed.). Princeton University Press.
