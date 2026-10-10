---
title: Causal Emergence
description: "Causal emergence is the claim that a coarse-grained macro description of a system can carry more causal information than its micro description, measured by effective information, with limits and links to the free-energy principle."
type: system
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - emergence
  - causality
  - information_theory
semantic_relations:
  - type: relates
    links:
      - "[[emergence]]"
      - "[[information_theory]]"
      - "[[circular_causality]]"
      - "[[free_energy_principle]]"
---

# Causal Emergence

## Overview

Causal emergence is a proposal, developed by Erik Hoel and collaborators, that macro-scale descriptions of a system can be causally stronger than the micro-scale ones from which they are coarse-grained. "Stronger" is made precise with an information-theoretic quantity, effective information. It is one formalisation of [[emergence]], and it is separate from philosophical strong emergence, which posits downward causation that is not reducible to micro dynamics.

## Formalism

Take a system with states $X$ and a transition probability matrix (TPM) $p(y \mid x)$. Effective information is the mutual information between cause and effect when the cause is set by an intervention to the maximum-entropy (uniform) distribution:

$EI = I(X_{\text{do}(\text{uniform})};\, Y)$.

It is high when each state leads reliably to a distinct next state (determinism) and when many different states are reachable (non-degeneracy). Noise lowers it by blurring outcomes, and degeneracy lowers it by letting many causes lead to the same effect.

A coarse-graining groups micro-states into macro-states and builds a macro TPM. Causal emergence occurs when $EI_{\text{macro}} > EI_{\text{micro}}$.

**Worked example.** Take four micro-states in which states 1-3 each move uniformly at random to one of states 1-3, and state 4 stays at state 4. Under uniform interventions, the micro EI is about 0.811 bits (computed directly from the TPM). Group states 1-3 into macro-state A and keep state 4 as B. The macro TPM is the identity, with EI equal to 1 bit. The macro description is more informative about effects because it discards the noise-driven transitions within the group.

## Criticisms and status

- The measure depends on the uniform intervention distribution, which may not match the system's natural dynamics, and different choices change the result.
- The macro scale gains because information is lost in the grouping, which critics argue shows that the macro is a better predictive description, not that it has independent causal power.
- Alternative frameworks (information decomposition, causal abstraction) treat the question differently, and no consensus exists. The approach remains an active research topic and should not be read as settled.

## Role in Active Inference

The connection is conceptual. The free-energy principle describes nested systems whose Markov blankets separate scales, and Active Inference models at each level treat the level below as noisy observations summarised into macro-states, which is a coarse-graining of the kind used above. Hierarchical generative models also choose a scale at which dynamics are most predictable. Proposals that link causal emergence to the principle use this resemblance; they have not produced a derivation, and the technical relation between effective information and free energy is an open question.

## Related concepts

- [[emergence]] — general phenomenon of macro-level pattern
- [[circular_causality]] — macro and micro constraining each other
- [[markov_blankets]] — boundaries defining scales in the free-energy principle
- [[information_theory]] — mutual information underlying effective information
- [[complex_systems]] — the settings where scale choice matters
- [[cellular_automata]] — standard test beds for coarse-graining
