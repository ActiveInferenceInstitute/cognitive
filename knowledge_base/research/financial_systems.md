---
title: Financial Systems
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
      - "[[knowledge_base/research/market_dynamics]]"
      - "[[knowledge_base/systems/risk_assessment]]"
      - "[[knowledge_base/systems/underwriting_principles]]"
      - "[[knowledge_base/free_energy_principle/applications/economics]]"
      - "[[knowledge_base/mathematics/expected_free_energy]]"
      - "[[knowledge_base/systems/network_theory]]"
      - "[[knowledge_base/systems/resilient_systems]]"
      - "[[knowledge_base/research/economic_policy]]"
---

# Financial Systems

## Overview

Financial systems allocate capital and manage risk through banks, markets, insurers and the infrastructure that connects them. Research addresses how individual institutions price and hold risk, how portfolios are constructed, and how the system as a whole can fail. This page treats the topic as risk-bearing decision-making and network stability, and leaves price formation to [[knowledge_base/research/market_dynamics]].

## Main questions

- **Portfolio choice.** Markowitz's mean-variance framework (1952) formalised diversification by trading expected return against variance. Later work added factor models, robust optimisation and constraints for estimation error.
- **Risk assessment.** Value-at-risk, expected shortfall, stress testing and scenario analysis. All depend on a model of tail events, which is the hardest part to estimate.
- **Systemic risk.** Failure of one institution transmitting to others through counterparty exposure, fire sales and common asset holdings. Haldane and May (2011) analysed banking networks with tools from ecology, showing that connectivity can stabilise a system against small shocks and amplify large ones.
- **Insurance and underwriting.** Pricing uncertain future liabilities from limited data.

## Connection to Active Inference

The relation is a conceptual mapping with early-stage modelling, and its first requirement is a clarification of terms. In expected free energy, "risk" is the divergence between the outcomes an agent predicts under a policy and the outcomes it prefers, and "ambiguity" is the expected uncertainty of observations given states. This risk is not the variance of returns that finance uses, and the two should not be conflated. A related but distinct line, risk-sensitive control, does connect to financial utility theory, because its objectives penalise variance or exponentiate costs. Active Inference can express risk attitudes through the form of the prior preferences.

For systemic stability, the framework suggests treating each institution as an agent whose model of the others' states drives its decisions, so that shared or correlated beliefs produce correlated actions and fragility. This is a reading of mechanisms already studied with network and agent-based models, not a competing result. There is no demonstration that Active Inference improves portfolio performance, risk estimates or stress-test design over standard methods. Practical use would require careful comparison on real data, with attention to the usual hazards of financial modelling: non-stationarity, data-snooping and model risk.

## In this repository

The insurance and underwriting thread is developed in [[knowledge_base/systems/underwriting_principles]] and [[active_inference_underwriting_learning_path]]. Related pages include [[knowledge_base/free_energy_principle/applications/economics]], [[knowledge_base/mathematics/expected_free_energy]], [[knowledge_base/systems/network_theory]], [[knowledge_base/systems/resilient_systems]] and [[knowledge_base/research/economic_policy]].

## References

- Markowitz, H. (1952). Portfolio selection. *Journal of Finance*, 7, 77-91.
- Haldane, A. G., & May, R. M. (2011). Systemic risk in banking ecosystems. *Nature*, 469, 351-355.

## Related Resources

- [[knowledge_base/research/market_dynamics]]
- [[knowledge_base/systems/risk_assessment]]
