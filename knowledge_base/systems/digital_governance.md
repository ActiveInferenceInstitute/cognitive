---
title: Digital Governance
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
      - "[[knowledge_base/systems/technology_ethics]]"
      - "[[knowledge_base/systems/social_systems]]"
      - "[[knowledge_base/systems/risk_assessment]]"
      - "[[knowledge_base/research/cognitive_security]]"
---

# Digital Governance

## Overview

Digital governance is the set of rules, roles, controls and oversight processes by which an organisation or a society decides how software and data systems may be built, deployed and changed, and who answers when they fail. For AI systems it covers regulation, internal management systems, documentation and technical controls. Active Inference does not require a separate governance regime: an agent built on it is an AI system and falls under the same instruments as any other. What the framework adds are some properties that make particular controls easier, discussed below. Those properties are design opportunities, not established compliance advantages.

## Instruments in Use

**Regulation.** The European Union's AI Act sorts systems into risk tiers and attaches obligations (documentation, human oversight, logging, post-market monitoring) to the higher tiers. Obligations depend on the use case, not on the algorithm, so an Active Inference controller in a medical device is regulated as a medical-device component.

**Management frameworks and standards.** The NIST AI Risk Management Framework organises work into Govern, Map, Measure and Manage functions, and ISO/IEC 42001 specifies an AI management system that an organisation can be audited against. Both are process-oriented and voluntary unless a contract or regulator requires them.

**Documentation practices.** Model cards (Mitchell et al. 2019) and datasheets for datasets record intended use, evaluation conditions and known limits so downstream users can judge fit.

**Technical controls.** Access control, tamper-evident audit logs, versioned models and configurations, staged rollout with rollback, and human approval for high-impact actions.

## Where Active Inference Helps and Where It Does Not

An agent's goals live in explicit objects, the preference vector $C$ and the priors $D$ and $E$, and its beliefs and policy probabilities are directly readable at every step. A reviewer can inspect and log those, and change a preference without retraining. This supports accountability for what the agent was trying to do. It does not make a large learned model interpretable, does not show the preferences are the right ones, and gives no guarantee about behaviour outside the modelled environment. Trust mechanisms still rest on testing, monitoring and independent review, and the main governance questions, who sets the preferences and who can override them, are institutional questions.

## In This Repository

This page connects to [[knowledge_base/systems/technology_ethics]], [[knowledge_base/systems/social_systems]], [[knowledge_base/systems/risk_assessment]], [[knowledge_base/research/cognitive_security]], [[knowledge_base/research/infohazard_management]] and [[knowledge_base/free_energy_principle/applications/ai_safety]]. The repository applies lightweight governance to itself: code is MIT-licensed and documentation is CC BY-NC-SA 4.0, and link and frontmatter checks (`cognitive-verify-links`, `cognitive-validate-docs`) run over the knowledge base. It ships no governance tooling for deployed agents, such as audit logging or policy enforcement.

## References

- Mitchell, M., et al. (2019). Model cards for model reporting. *Proceedings of the Conference on Fairness, Accountability, and Transparency*.
- National Institute of Standards and Technology (2023). *Artificial Intelligence Risk Management Framework (AI RMF 1.0)*.

## Related Resources

- [[knowledge_base/systems/technology_ethics]]
- [[knowledge_base/systems/social_systems]]
