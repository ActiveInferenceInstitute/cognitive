---
title: Weak Emergence
description: The view that higher-level patterns are derivable from lower-level dynamics yet only by simulation, with Bedau's definition, cellular automaton examples, and its fit with free energy accounts of self-organisation.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - emergence
  - complex_systems
  - philosophy_of_science
semantic_relations:
  - type: relates
    links:
      - "[[emergence]]"
      - "[[strong_emergence]]"
      - "[[self_organization]]"
      - "[[cellular_automata]]"
      - "[[complex_systems]]"
---

# Weak Emergence

## Overview
Weak emergence describes macro-level patterns that are fully determined by micro-level dynamics but cannot be predicted by any shortcut. Bedau (1997) defined a macrostate as weakly emergent when it is derivable from the system's microdynamics and external conditions, but only by simulating the system step by step. Such patterns are autonomous in practice, because they must be observed or simulated, yet they involve no new fundamental laws and no causal power beyond that of the parts.

## Key features
- **Derivable but incompressible.** The surprise is epistemic. The micro-rules fix the outcome, but the outcome cannot be compressed into a short calculation.
- **Standard examples.** Gliders in Conway's Game of Life follow from a simple local rule, but their existence, speed and interactions are found by running the automaton (see [[cellular_automata]]). Flocking, traffic jams, convection cells and ant trails are other cases usually described this way.
- **Contrast with strong emergence.** Chalmers (2006) classed weak emergence as the unproblematic kind, compatible with physicalism, whereas [[strong_emergence]] would demand new laws.
- **Scientific use.** Because compressed descriptions often exist at coarse scales, higher-level models can be predictive and explanatory even though the underlying derivation is always available in principle. Causal emergence work asks when a coarse-grained description carries more information about outcomes than the micro-description (see [[causal_emergence]]).

## Relation to Active Inference
Weak emergence is the natural reading of the free energy principle's treatment of [[self_organization]]. Coupled microscopic dynamics, partitioned by a Markov blanket, yield system-level behaviour that appears to minimise surprise, and the system-level description is a coarse-graining of the lower-level one. This is a principled interpretive fit, not a theorem that excludes stronger readings. How far such coarse-grained descriptions have causal significance of their own remains contested.

## Related concepts
- [[emergence]] — the umbrella notion this page narrows
- [[strong_emergence]] — the contrasting claim of non-deducible properties
- [[self_organization]] — order arising from local interactions
- [[cellular_automata]] — the standard testbed for simulation-only derivation
- [[causal_emergence]] — information-theoretic account of macro-level causal structure
- [[complex_systems]] — domain where such patterns are studied

## References
- Bedau, M. A. (1997). Weak emergence. Philosophical Perspectives, 11, 375-399.
- Chalmers, D. J. (2006). Strong and weak emergence. In P. Clayton & P. Davies (Eds.), The Re-Emergence of Emergence (pp. 244-256). Oxford University Press.
