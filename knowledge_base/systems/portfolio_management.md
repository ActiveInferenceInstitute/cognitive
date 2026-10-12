---
title: Portfolio Management
description: "Portfolio management selects and balances a collection of risks or assets under uncertainty, covering diversification, correlation, risk aggregation and capital limits as applied to insurance underwriting and Active Inference."
type: system
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - risk
  - insurance
  - decision_making
semantic_relations:
  - type: relates
    links:
      - "[[risk_assessment]]"
      - "[[underwriting_principles]]"
      - "[[resource_allocation]]"
      - "[[active_inference]]"
wikidata: Q16167513
---

# Portfolio Management

## Overview

Portfolio management is the practice of choosing, sizing and monitoring a collection of exposures so that the combined outcome meets objectives for return and tolerance of loss. In the underwriting pages of this repo the portfolio is a book of insured risks; in finance it is a set of assets. In both, the central fact is that the risk of the whole differs from the sum of the risks of the parts, because outcomes of the parts are dependent.

## Core ideas

- **Diversification**: combining imperfectly correlated exposures reduces total variability. For two assets weighted equally with volatilities 20% and 10%, the portfolio volatility is 15% if they are perfectly correlated, about 11.2% if uncorrelated and about 8.7% if the correlation is -0.5 (computed from the variance formula $\sigma_p^2 = w_1^2\sigma_1^2 + w_2^2\sigma_2^2 + 2 w_1 w_2 \rho \sigma_1 \sigma_2$).
- **Limits of diversification**: for $n$ equally weighted exposures each with volatility 20% and pairwise correlation 0.3, portfolio volatility is about 16.1% at $n=2$, 12.2% at $n=10$ and 11.1% at $n=100$, approaching $\sigma\sqrt{\rho} \approx 11.0\%$. Beyond a point, only the common (systematic) component remains.
- **Mean-variance optimisation** (Markowitz, 1952): choose weights that minimise variance for a target expected return. Results are sensitive to estimation error in expected returns and correlations.
- **Risk aggregation and tail dependence**: in insurance, correlated catastrophes (a regional flood or storm) make losses cluster, so averages understate capital needs. Dependence in the tail is often larger than a linear correlation suggests.
- **Concentration and accumulation limits**: caps by geography, peril, counterparty or line of business keep any one scenario survivable.
- **Risk measures**: value at risk and expected shortfall quantify the loss tail. Expected shortfall is coherent, while value at risk is not generally subadditive.
- **Reinsurance and hedging** transfer part of the exposure to others, trading expected profit for stability.

## Role in Active Inference

The connection is analogical and in part an application. Portfolio selection is decision-making under uncertainty with a preference over outcomes, which maps onto expected free energy with its balance of risk (divergence of predicted outcomes from preferred ones) and ambiguity. A risk-averse underwriter is modelled by strongly weighted prior preferences against loss, and diversification as the choice of policies that keep outcome distributions close to preferred ranges. Learning about correlations from claims data is Bayesian parameter learning. No standard Active Inference portfolio model exists, and results from the free-energy literature do not replace the actuarial and financial methods described above.

## Related concepts

- [[risk_assessment]] — estimating individual exposures before they are combined
- [[underwriting_principles]] — selection principles applied to each risk
- [[resource_allocation]] — dividing limited capital across options
- [[expected_free_energy]] — the repo's risk-plus-ambiguity objective
- [[game_theory]] — strategic aspects of markets and reinsurance
- [[robustness]] — surviving adverse scenarios
