---
title: Econophysics
description: The application of concepts and methods from statistical physics, such as power laws, random walks and statistical mechanics, to the analysis of financial markets and the distribution of wealth and income.
type: concept
status: stable
created: 2026-10-12
updated: 2026-10-12
tags:
  - statistical_physics
  - economics
  - power_laws
  - financial_markets
semantic_relations:
  - type: relates
    links:
      - "[[statistical_physics]]"
      - "[[power_laws]]"
      - "[[random_walk]]"
      - "[[market_dynamics]]"
      - "[[philosophy_of_economics]]"
      - "[[agent_based_modeling]]"
      - "[[sociophysics]]"
---

# Econophysics

## Overview
Econophysics applies the tools of statistical physics to economic and financial data. Markets are large systems of interacting agents that produce abundant data, and some of their statistical regularities resemble those of physical systems with many degrees of freedom. Mantegna and Stanley (1999) wrote one of the first textbooks on the subject, centred on correlations and complexity in finance. The field is closely related to [[sociophysics]], which applies the same style of modelling to non-economic social behaviour.

## Main topics
- **Statistical properties of asset prices.** Cont (2001) compiles stylised facts that recur across markets and instruments, including heavy-tailed return distributions, volatility clustering and the absence of simple linear autocorrelation of returns. Random-walk models of price are the baseline against which these departures are described (see [[random_walk]]).
- **Power laws.** Gabaix (2009) reviews how power-law distributions arise for quantities such as firm size, city size and large financial returns, and discusses mechanisms that generate them (see [[power_laws]]).
- **Wealth and income distributions.** Yakovenko and Rosser (2009) review statistical-mechanics models of money, wealth and income. In these, money is conserved in pairwise exchanges between agents, in analogy with energy in collisions between molecules, which leads to exponential distributions for the bulk of the population, while the upper tail of income and wealth is better described by a power law.
- **Agent-based models** of markets, in which many simple traders interact, are used to test which mechanisms reproduce the stylised facts (see [[agent_based_modeling]]).

## Critique
The same observation of a power law or heavy tail can come from different mechanisms, so fitting a distribution does not establish its cause. Conserved-money models omit production, credit and strategic behaviour by construction, so they are best treated as descriptive baselines and as sources of testable constraints, not as complete theories of an economy.

## Relation to Active Inference
The two approaches differ in their treatment of agents. Econophysics usually takes agents as simple, randomly interacting units and studies the aggregate statistics, whereas Active Inference gives each agent a generative model and preferences and derives behaviour from free energy minimisation. None of the sources cited here applies the free energy principle to economic data, so this page asserts no link between econophysics results and Active Inference; see [[market_dynamics]] and [[philosophy_of_economics]] for related material. Any claim that markets minimise free energy would be an interpretation, not a result of econophysics.

## Related concepts
- [[power_laws]] — the characteristic distributions of the field
- [[random_walk]] — the baseline model of prices
- [[statistical_physics]] — the source of methods
- [[sociophysics]] — the parallel programme for social behaviour
- [[market_dynamics]] — economic dynamics from a modelling viewpoint

## References
- Mantegna, R. N., & Stanley, H. E. (1999). An Introduction to Econophysics: Correlations and Complexity in Finance. Cambridge University Press. https://doi.org/10.1017/CBO9780511755767
- Cont, R. (2001). Empirical properties of asset returns: stylized facts and statistical issues. Quantitative Finance, 1(2), 223-236. https://doi.org/10.1080/713665670
- Gabaix, X. (2009). Power laws in economics and finance. Annual Review of Economics, 1, 255-294. https://doi.org/10.1146/annurev.economics.050708.142940
- Yakovenko, V. M., & Rosser, J. B. (2009). Colloquium: Statistical mechanics of money, wealth, and income. Reviews of Modern Physics, 81(4), 1703-1725. https://doi.org/10.1103/RevModPhys.81.1703
