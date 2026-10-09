---
title: Security Evolution
type: knowledge_base
status: stable
created: 2026-02-07
updated: 2026-10-09
tags:
  - research
  - active_inference
semantic_relations:
  - type: relates_to
    links:
      - "[[knowledge_base/research/cognitive_security]]"
      - "[[knowledge_base/systems/resilient_systems]]"
      - "[[knowledge_base/free_energy_principle/biology/immunology]]"
      - "[[knowledge_base/biology/immunology]]"
      - "[[knowledge_base/biology/evolutionary_game_theory]]"
      - "[[knowledge_base/systems/fault_tolerance]]"
      - "[[knowledge_base/research/infohazard_management]]"
---

# Security Evolution

## Overview

Living systems have faced attackers for as long as there have been hosts: parasites, pathogens, predators and cheats. The mechanisms that evolved in response form a long record of defensive design, and they inform the design of artificial security. This page surveys that record and states how far Active Inference applies to it.

## Biological defence

- **Immunity.** Vertebrates combine innate recognition of conserved microbial patterns with adaptive immunity that generates and selects receptors against specific antigens and retains memory. Distinguishing self from non-self, and tolerating harmless non-self, are the central problems.
- **Prokaryotic defence.** Bacteria use restriction-modification systems and CRISPR-Cas, an adaptive system that stores fragments of past viral infections.
- **Social immunity.** Group-living insects reduce disease spread through grooming, waste removal and behavioural changes, a pattern reviewed by Cremer, Armitage and Schmid-Hempel (2007).
- **Coevolutionary arms races.** Van Valen's Red Queen hypothesis (1973) and Dawkins and Krebs (1979) describe how adaptation on one side drives counter-adaptation on the other, so that neither gains lasting advantage.

## Artificial parallels

Computer security has borrowed several of these ideas, including artificial immune systems for anomaly detection, diversity as a defence (software variants that do not share the same flaw), and moving-target defence. The analogy has limits. Biological defences are tuned by selection over many generations against a stochastic threat, while attackers on computer systems are intelligent and adapt within the same episode.

## Connection to Active Inference

The relation is an interpretive framework with some formal support, not a settled explanation of any specific defence. Under the free energy principle a persistent system maintains a boundary (a Markov blanket) and keeps its states within a viable range. Threats are events that push the system toward states it does not expect to occupy, and defences are mechanisms that restore expected states or avoid them. On this reading the immune system can be described as an inferential process that estimates the presence of threats from noisy signals and acts on that estimate. The repository develops this view in [[knowledge_base/free_energy_principle/biology/immunology]].

Adversarial settings add a feature the single-agent formalism lacks: the opponent is itself an adaptive agent whose behaviour must be modelled. Multi-agent extensions of Active Inference address this in principle, but models of attacker-defender co-adaptation are at an early stage, and the Red Queen dynamics above are explained well by existing evolutionary game theory without invoking the free energy principle.

## In this repository

See [[knowledge_base/biology/immunology]] for the underlying biology, [[knowledge_base/biology/evolutionary_game_theory]] for the arms-race formalism, [[knowledge_base/systems/fault_tolerance]] and [[knowledge_base/systems/risk_assessment]] for engineering counterparts, and [[knowledge_base/research/infohazard_management]] for information-level threats.

## References

- Van Valen, L. (1973). A new evolutionary law. *Evolutionary Theory*, 1, 1-30.
- Dawkins, R., & Krebs, J. R. (1979). Arms races between and within species. *Proceedings of the Royal Society B*, 205, 489-511.
- Cremer, S., Armitage, S. A. O., & Schmid-Hempel, P. (2007). Social immunity. *Current Biology*, 17, R693-R702.

## Related Resources

- [[knowledge_base/research/cognitive_security]]
- [[knowledge_base/systems/resilient_systems]]
