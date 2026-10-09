---
title: Superintelligence Research
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
      - "[[knowledge_base/cognitive/active_inference]]"
      - "[[knowledge_base/research/agi_development]]"
      - "[[knowledge_base/free_energy_principle/applications/ai_safety]]"
      - "[[knowledge_base/cognitive/preference_learning]]"
      - "[[knowledge_base/systems/technology_ethics]]"
---

# Superintelligence Research

## Overview

Superintelligence refers to a hypothetical agent whose cognitive performance greatly exceeds that of humans across most domains of interest. The term was given its current prominence by Nick Bostrom's 2014 book *Superintelligence: Paths, Dangers, Strategies*. The idea has an older root in I. J. Good's 1965 remark about an "ultraintelligent machine" that could design better machines. No such system exists, and the field mixes formal arguments, philosophical analysis and informed disagreement about plausibility and timing. This page describes the concepts and does not take a position on whether or when it will occur.

## Main questions

- **Pathways.** Whether capable systems would arise from scaling learning methods, whole-brain emulation, biological enhancement, or collectives of agents, and whether progress would be gradual or abrupt.
- **Recursive self-improvement.** Whether a system could improve its own design in a loop that accelerates. This is an argument, not an observed phenomenon, and its premises (for example that returns to intelligence do not diminish quickly) are contested.
- **Goals and control.** The orthogonality thesis holds that intelligence and final goals vary independently. The instrumental convergence argument (associated with Omohundro and Bostrom) holds that many goals favour sub-goals such as resource acquisition and self-preservation. Both are philosophical claims that critics dispute in scope and strength.
- **Alignment and oversight.** How to specify or learn human values, keep systems correctable, and verify behaviour. Proposals include cooperative inverse reinforcement learning, in which the agent is uncertain about the human's objective, and the approach in Stuart Russell's *Human Compatible* (2019).

## Connection to Active Inference

The connection is an analogy supported by a small amount of theory, not an established result. Three points are worth separating.

First, "unbounded inference" has a precise idealised reading: an agent that performs exact Bayesian inference and expected free energy minimisation with no computational limit. Real Active Inference agents rely on approximate inference, and approximation error is the reason such idealisations do not describe actual behaviour.

Second, goals in the framework are prior preferences over outcomes. An agent that must also keep its own states within viable bounds has a built-in drive toward self-maintenance, which resembles the instrumental-convergence argument. Whether this makes such agents more or less controllable is an open question, and arguments in both directions have been made.

Third, the epistemic term of expected free energy gives agents intrinsic information-seeking. Some authors suggest this changes the alignment problem relative to pure reward maximisation, for instance by making preference uncertainty explicit. This is a hypothesis awaiting formal and empirical work.

## In this repository

Safety-oriented material is in [[knowledge_base/free_energy_principle/applications/ai_safety]]. The agent formulation is in [[knowledge_base/mathematics/expected_free_energy]], and learning of goals from data is covered in [[knowledge_base/cognitive/preference_learning]]. Ethical framing of technology appears in [[knowledge_base/systems/technology_ethics]].

## References

- Good, I. J. (1965). Speculations concerning the first ultraintelligent machine. *Advances in Computers*, 6.
- Bostrom, N. (2014). *Superintelligence: Paths, Dangers, Strategies*. Oxford University Press.
- Russell, S. (2019). *Human Compatible*. Viking.

## Related Resources

- [[knowledge_base/cognitive/active_inference]]
- [[knowledge_base/research/agi_development]]
