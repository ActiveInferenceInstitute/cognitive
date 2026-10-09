---
title: Market Dynamics
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
      - "[[knowledge_base/research/economic_policy]]"
      - "[[knowledge_base/research/financial_systems]]"
      - "[[knowledge_base/free_energy_principle/applications/economics]]"
      - "[[knowledge_base/mathematics/game_theory]]"
      - "[[knowledge_base/cognitive/multi_agent_active_inference]]"
      - "[[knowledge_base/cognitive/collective_behavior]]"
      - "[[knowledge_base/cognitive/social_learning]]"
      - "[[knowledge_base/systems/network_theory]]"
      - "[[knowledge_base/systems/underwriting_principles]]"
---

# Market Dynamics

## Overview

Market dynamics concerns how prices, volumes and beliefs evolve in markets, and how the rules of trading shape that process. Two classic positions frame the field. Hayek (1945) argued that prices aggregate dispersed private knowledge that no single agent possesses. The efficient market hypothesis (Fama, 1970) states that prices reflect available information, in versions of different strength. Empirical work has documented both substantial informational efficiency and persistent departures from it, including bubbles, momentum, and excess volatility.

## Main questions

- **Price formation.** How private information enters prices through trading. Kyle's (1985) model of an informed trader and a market maker, and Glosten and Milgrom's (1985) sequential trade model, are standard microstructure results, in which the market maker's quotes are Bayesian updates on order flow.
- **Market microstructure.** The limit order book, liquidity, bid-ask spreads and the effect of trading rules and latency.
- **Beliefs and expectations.** Keynes's "beauty contest" analogy (1936): when agents try to anticipate others' expectations, prices can depend on higher-order beliefs and not only on fundamentals.
- **Bubbles and crashes.** Herding, leverage and feedback between price changes and trading behaviour.
- **Agent-based market models.** Heterogeneous agents following simple rules, used to reproduce stylised facts such as fat-tailed returns and volatility clustering.

## Connection to Active Inference

The connection is an analogy with some early modelling. Market microstructure is already Bayesian: the Kyle and Glosten-Milgrom models are belief-updating models, and prices act as a public signal about what others know. Active Inference adds a way to describe how heterogeneous agents combine belief updating with action selection that values both profit and information, and to describe higher-order beliefs about other traders, which parallels the beauty-contest idea. The phrase "collective belief updating" describes the Hayekian picture in this vocabulary.

No evidence yet shows that this vocabulary yields better price forecasts, or explains stylised facts, more economically than existing agent-based and microstructure models. Any claim that prices literally minimise a collective free energy is a metaphor until a specific market model is formulated and tested. The most useful near-term contribution is to represent agents who explore and exploit under uncertainty within one framework, and then to check the model's output against market data.

## In this repository

Related pages are [[knowledge_base/free_energy_principle/applications/economics]], [[knowledge_base/mathematics/game_theory]], [[knowledge_base/cognitive/multi_agent_active_inference]], [[knowledge_base/cognitive/collective_behavior]], [[knowledge_base/cognitive/social_learning]], [[knowledge_base/systems/network_theory]] and [[knowledge_base/systems/underwriting_principles]]. A study path is [[active_inference_economic_learning_path]].

## References

- Hayek, F. A. (1945). The use of knowledge in society. *American Economic Review*, 35, 519-530.
- Fama, E. F. (1970). Efficient capital markets: a review of theory and empirical work. *Journal of Finance*, 25, 383-417.
- Kyle, A. S. (1985). Continuous auctions and insider trading. *Econometrica*, 53, 1315-1335.
- Glosten, L. R., & Milgrom, P. R. (1985). Bid, ask and transaction prices in a specialist market with heterogeneously informed traders. *Journal of Financial Economics*, 14, 71-100.

## Related Resources

- [[knowledge_base/research/economic_policy]]
- [[knowledge_base/research/financial_systems]]
