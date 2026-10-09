---
title: Infohazard Management
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
      - "[[knowledge_base/research/security_evolution]]"
      - "[[knowledge_base/cognitive/precision_weighting]]"
      - "[[knowledge_base/systems/risk_assessment]]"
---

# Infohazard Management

## Overview

An information hazard, or infohazard, is a risk that arises from the dissemination or possession of true information. The standard typology is Nick Bostrom's "Information Hazards: A Typology of Potential Harms from Knowledge" (2011). It distinguishes, among others, data hazards (specific dangerous facts, such as a recipe), idea hazards (general insights that enable harm), attention hazards (harm from drawing notice to something already public), template hazards (a pattern that others can imitate), signaling hazards, and evocation hazards (information that causes distress by its nature). Infohazard management is the practice of deciding who should hold, share or publish such information, and how.

## Practices and trade-offs

The concept is used mainly in biosecurity, AI safety and cybersecurity. Established practices include:

- **Need-to-know and tiered access**, as in classified work.
- **Coordinated disclosure**, in which a vulnerability is reported privately to its maintainers and published after a fixed window.
- **Staged release**, as when AI developers have withheld or delayed model weights or details; the 2019 staged release of the GPT-2 language model is a frequently cited case.
- **Dual-use review**, applied to life-science research whose methods could enable misuse.

The central difficulty is that secrecy has costs. Concealment slows verification and correction, can draw attention to what is hidden, and may fail once information leaks. Openness supports defence and scientific progress but can lower barriers for misuse. There is no general rule; judgements depend on counterfactual availability, severity, and who benefits. Claims in this area are often hard to test, and reasonable people disagree about thresholds.

## Connection to Active Inference

The relation is partly formal and partly an analogy. In Active Inference an agent values information through the epistemic term of expected free energy, which is an expected information gain. Expected information gain is a mutual information and is never negative, so for an idealised Bayesian agent acting for its own ends, more evidence does not lower expected epistemic value. Infohazards therefore arise outside that idealisation: from third parties who use the information, from agents with bounded computation or fixed commitments, from strategic interaction, or from information that changes an agent's own preferences or emotional state. This framing is a useful way to ask where a harm comes from; it is not an established theory of infohazards.

A second link concerns the vulnerability of inference itself. A system that updates beliefs by weighting evidence according to estimated reliability (precision) can be misled by inputs crafted to exploit that weighting. Adversarial examples in machine learning, first reported for neural networks by Szegedy and colleagues in 2013, are the best-documented technical instance. Whether human belief manipulation follows the same mechanism is an active modelling question and is treated in [[knowledge_base/research/cognitive_security]].

## In this repository

The security cluster is [[knowledge_base/research/cognitive_security]] and [[knowledge_base/research/security_evolution]]. Precision weighting is described in [[knowledge_base/cognitive/precision_weighting]]. Risk framing is in [[knowledge_base/systems/risk_assessment]]. Learning material is in [[active_inference_security_learning_path]].

## References

- Bostrom, N. (2011). Information hazards: a typology of potential harms from knowledge. *Review of Contemporary Philosophy*, 10, 44-79.
- Szegedy, C., et al. (2013). Intriguing properties of neural networks. arXiv:1312.6199.

## Related Resources

- [[knowledge_base/research/cognitive_security]]
- [[knowledge_base/research/security_evolution]]
