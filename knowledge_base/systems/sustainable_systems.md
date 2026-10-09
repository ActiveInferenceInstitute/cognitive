---
title: Sustainable Systems
type: knowledge_base
status: stable
created: 2026-02-07
updated: 2026-10-09
tags:
  - systems
  - active_inference
semantic_relations:
  - type: relates_to
    links:
      - "[[knowledge_base/systems/resilience_thinking]]"
      - "[[knowledge_base/systems/ecosystem_management]]"
      - "[[knowledge_base/systems/adaptive_comanagement]]"
      - "[[knowledge_base/cognitive/homeostatic_regulation]]"
---

# Sustainable Systems

## Overview

A sustainable system meets present needs without eroding the capacity to meet future ones: it keeps resource use within regenerative limits and keeps functioning under shock. The design problem spans technical choices (energy, materials, supply chains), management (how decisions adapt as evidence arrives) and governance (who decides and who bears costs). The established tools come from resilience science, adaptive management and industrial ecology. Active Inference offers a way of describing the regulation problem, and its use for resource management or circular-economy optimisation is a proposal without substantial field validation.

## Established Concepts

**Resilience.** Holling's distinction between engineering resilience (speed of return to equilibrium) and ecological resilience (the magnitude of disturbance a system absorbs before shifting to another regime) frames much of the field ([[knowledge_base/systems/resilience_thinking]]).

**Adaptive management.** Policies are treated as experiments: act, monitor, learn, revise. Its difficulty is that learning is slow and costly when the system is large ([[knowledge_base/systems/adaptive_comanagement]]).

**Planetary boundaries and leverage points.** Rockstrom and colleagues (2009) proposed limits for Earth-system processes, and Meadows's leverage points rank where intervention changes system behaviour most.

**Circular economy.** Designing to reuse, repair and recycle keeps materials in use. Reported gains depend on system boundaries, since recycling that needs more energy than virgin production, or rebound effects that raise total consumption, can cancel the benefit.

## Connection to Active Inference

The most direct mapping is homeostatic and allostatic regulation. An agent that keeps key variables inside preferred bounds, such as a reservoir level, soil carbon or a stock of materials, acts to keep its observations near preferred values while resolving uncertainty about how the system responds ([[knowledge_base/cognitive/homeostatic_regulation]], [[knowledge_base/cognitive/allostatic_control]]). Adaptive management maps onto the trade-off between exploiting current knowledge and probing for information. Value choices enter through the preference vector: the agent will pursue what its preferences specify, so deciding what to sustain, for whom, and over what horizon cannot be delegated to the model. These are correspondences in description. They do not by themselves show that the framework produces better policies than established decision-analysis methods.

## In This Repository

Related pages: [[knowledge_base/systems/resilient_systems]], [[knowledge_base/systems/circular_causality]], [[knowledge_base/cognitive/resource_management]], [[knowledge_base/BioFirm/socioeconomic_active_inference]] and [[knowledge_base/BioFirm/ecological_active_inference]]. In code, `HomeostaticInference` in `code/tools/src/models/active_inference/homeostatic.py` is the nearest piece: it is configured with state spaces and a target, and two control modes shape the policy prior. `HomeostaticControl` favours beliefs near a target, and `AdaptiveControl` blends target tracking with an exploration term based on prediction error. It is a small discrete controller for a single regulated variable, not a model of a resource system or an economy.

## References

- Holling, C. S. (1973). Resilience and stability of ecological systems. *Annual Review of Ecology and Systematics*.
- Rockstrom, J., et al. (2009). A safe operating space for humanity. *Nature*.
- Meadows, D. (1999). *Leverage Points: Places to Intervene in a System*. The Sustainability Institute.

## Related Resources

- [[knowledge_base/systems/resilience_thinking]]
- [[knowledge_base/systems/ecosystem_management]]
