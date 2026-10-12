---
title: Financial Mathematics
description: The use of probability, stochastic calculus and optimisation to model asset prices and to price derivative securities, built on the no-arbitrage principle and the Black-Scholes-Merton model.
type: mathematical_concept
status: stable
created: 2026-10-12
updated: 2026-10-12
tags:
  - finance
  - stochastic_calculus
  - pricing
semantic_relations:
  - type: relates
    links:
      - "[[stochastic_calculus]]"
      - "[[ito_calculus]]"
      - "[[brownian_motion]]"
      - "[[stochastic_processes]]"
      - "[[risk_preference]]"
      - "[[financial_systems]]"
---

# Financial Mathematics

## Overview
Financial mathematics, also called mathematical finance, builds quantitative models of financial markets. Its two largest themes are the pricing of derivative securities, whose payoff depends on the future price of another asset, and portfolio choice under uncertainty. Its tools come mainly from probability and [[stochastic_calculus]]. A starting point is Bachelier's 1900 thesis, which modelled price movements as Brownian motion. Textbook accounts include Shreve (2004).

## Asset prices and no-arbitrage
The standard continuous-time model treats a stock price $S_t$ as geometric Brownian motion,
$$dS_t=\mu S_t\,dt+\sigma S_t\,dW_t,$$
where $\mu$ is the drift, $\sigma$ the volatility and $W_t$ a Wiener process (see [[brownian_motion]] and [[ito_calculus]]). Alongside it sits a riskless asset growing at rate $r$. An **arbitrage** is a trading strategy that starts from zero capital, never loses and has positive probability of gain. Pricing theory assumes there is none.

## The Black-Scholes-Merton model
Black and Scholes (1973) and Merton (1973) showed that a European option can be replicated by continuously rebalancing a portfolio of the stock and the riskless asset, so that its price $V(S,t)$ is fixed by the absence of arbitrage. The price satisfies the partial differential equation
$$\frac{\partial V}{\partial t}+\tfrac12\sigma^2S^2\frac{\partial^2V}{\partial S^2}+rS\frac{\partial V}{\partial S}-rV=0 .$$
Notably the drift $\mu$ does not appear. For a European call with strike $K$ and maturity $T$ the solution is
$$C=S\,\Phi(d_1)-Ke^{-rT}\Phi(d_2),\quad d_{1,2}=\frac{\ln(S/K)+(r\pm\tfrac12\sigma^2)T}{\sigma\sqrt T},$$
where $\Phi$ is the standard normal distribution function.

## Risk-neutral pricing
Harrison and Pliska (1981) put the theory on a martingale footing. In the fundamental theorem of asset pricing, which holds in precise form under technical conditions, absence of arbitrage corresponds to the existence of an equivalent probability measure $\mathbb{Q}$ under which discounted asset prices are martingales, and the price of a payoff $X$ at time $T$ is
$$V_0=e^{-rT}\,\mathbb{E}_{\mathbb{Q}}[X].$$
This converts a pricing problem into an expectation, which can be computed analytically, by solving a PDE, or by Monte Carlo simulation.

## Limits of the model
The Black-Scholes-Merton model assumes constant volatility, continuous trading and no transaction costs. Extensions allow stochastic volatility, jumps and incomplete markets, in which a unique risk-neutral price no longer exists.

## Relation to Active Inference
The shared element is mathematical, namely stochastic differential equations and expectations under changed measures. There is no established derivation that links option pricing to Active Inference. Treatments of risk in decision making inside the framework, such as risk-sensitive formulations (see [[risk_preference]]), parallel the role of risk attitudes in portfolio theory without being the same construction.

## Related concepts
- [[stochastic_calculus]] — the calculus on which continuous-time models rest
- [[ito_calculus]] — integration against Brownian motion
- [[brownian_motion]] — the driving noise
- [[stochastic_processes]] — random evolution in time
- [[risk_preference]] — attitudes to risk
- [[financial_systems]] — applications to markets in this knowledge base

## References
- Bachelier, L. (1900). Théorie de la spéculation. Annales scientifiques de l'École normale supérieure, 3e série, 17, 21-86. https://doi.org/10.24033/asens.476
- Black, F., & Scholes, M. (1973). The pricing of options and corporate liabilities. Journal of Political Economy, 81(3), 637-654. https://doi.org/10.1086/260062
- Merton, R. C. (1973). Theory of rational option pricing. The Bell Journal of Economics and Management Science, 4(1), 141-183. https://doi.org/10.2307/3003143
- Harrison, J. M., & Pliska, S. R. (1981). Martingales and stochastic integrals in the theory of continuous trading. Stochastic Processes and their Applications, 11(3), 215-260. https://doi.org/10.1016/0304-4149(81)90026-0
- Shreve, S. E. (2004). Stochastic Calculus for Finance II: Continuous-Time Models. Springer.
