---
title: System Dynamics
description: Modelling method developed by Jay Forrester that represents systems as stocks, flows, feedback loops and delays, used to study policy and long-term behaviour, and its modest link to Active Inference.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - system_dynamics
  - feedback
  - simulation
semantic_relations:
  - type: relates
    links:
      - "[[feedback_loops]]"
      - "[[systems_theory]]"
      - "[[dynamical_systems]]"
      - "[[differential_equations]]"
      - "[[cybernetics]]"
---

# System Dynamics

## Overview
System dynamics is a method for understanding the behaviour of complex systems over time, developed by Jay Forrester at MIT in the 1950s and 1960s, first for industrial management and later for urban and global problems. It holds that the behaviour of a system arises mainly from its feedback structure, so that explanations should focus on the loops and delays within the system, not on external events. Models are built by identifying the key accumulations and the causal links that change them, then simulated to examine consequences.

## Building blocks
- **Stocks** are accumulations, such as inventory, population or capital. Their value at any time is the integral of the net flow: $\dfrac{dS}{dt}=\text{inflow}-\text{outflow}$.
- **Flows** change stocks, and they depend on stocks and parameters through decision rules.
- **Feedback loops** are closed chains of influence. Reinforcing loops amplify change and give exponential growth, whereas balancing loops seek a goal and give stabilisation.
- **Delays and nonlinearities** make the response lag behind its causes, and the combination of delays with feedback produces overshoot, oscillation and counterintuitive policy effects.

Causal loop diagrams are used to sketch the structure, and stock-and-flow diagrams to specify the equations that are solved numerically, typically as coupled ordinary differential equations or their discrete-time approximations.

## Applications
Well-known applications include models of supply chains, epidemics, resource depletion and urban growth. The World3 model behind the 1972 report The Limits to Growth is the best-known and most contested example. Critics point to aggregated variables and hard-to-justify parameters, and practitioners respond by emphasising that models are tools for exploring dynamics and testing policies, not for forecasting.

## Relation to Active Inference
The link is modest and indirect. Both approaches treat behaviour as arising from circular causation and describe systems with differential equations, and both draw on the cybernetic tradition. System dynamics models are however usually aggregate and deterministic, are specified by the modeller, and do not include an agent that infers hidden states. A feedback structure in a system dynamics model could in principle describe the environment in which an Active Inference agent acts.

## Related concepts
- [[feedback_loops]] — the structural element that drives behaviour
- [[systems_theory]] — general theory of organised wholes
- [[dynamical_systems]] — mathematical basis for stock and flow equations
- [[differential_equations]] — formal language of the simulation models
- [[cybernetics]] — historical source of feedback thinking
- [[complex_systems]] — the field in which delayed, nonlinear feedback is studied

## References
- Forrester, J. W. (1961). Industrial Dynamics. MIT Press.
- Sterman, J. D. (2000). Business Dynamics: Systems Thinking and Modeling for a Complex World. Irwin/McGraw-Hill.
- Meadows, D. H. (2008). Thinking in Systems: A Primer. Chelsea Green Publishing.
