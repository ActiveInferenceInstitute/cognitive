---
title: Granger Causality
description: A statistical notion of predictive causality in time series, its definition by improved forecasting, its limits, and its contrast with dynamic causal modelling of effective connectivity.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - time_series
  - causality
  - connectivity
semantic_relations:
  - type: relates
    links:
      - "[[causal_inference]]"
      - "[[connectivity]]"
      - "[[neuroimaging]]"
      - "[[mutual_information]]"
      - "[[stochastic_processes]]"
---

# Granger Causality

## Overview
Granger causality is a criterion for whether one time series helps predict another. A series $X$ is said to Granger-cause a series $Y$ if the past of $X$ contains information that improves the prediction of $Y$ beyond what is available from the past of $Y$ and other relevant variables. The idea was formalised by Granger (1969), building on a suggestion by Wiener, and was developed in econometrics before spreading to neuroscience.

## Formulation
In the standard linear version, one fits two autoregressive models of $Y_t$. The restricted model uses only past values of $Y$, $Y_t=\sum_{k=1}^{p}a_kY_{t-k}+\varepsilon_t$. The full model adds past values of $X$, $Y_t=\sum_{k=1}^{p}a_kY_{t-k}+\sum_{k=1}^{p}b_kX_{t-k}+\varepsilon'_t$. $X$ Granger-causes $Y$ if the full model has significantly smaller residual variance, usually assessed with an F-test of the hypothesis $b_1=\dots=b_p=0$. The strength is often summarised as the log ratio of the two residual variances. Multivariate versions condition on additional series, and spectral versions decompose the influence by frequency.

## Interpretation and limits
The notion is about prediction, not intervention. Results are reliable only under conditions that are easy to violate:
- **Omitted variables**: a common driver omitted from the model produces apparent causality between its effects.
- **Measurement noise and downsampling**: noise differences between series, or sampling slower than the dynamics, can create or remove apparent directional influence. In functional MRI, regional variation in the haemodynamic response is a recognised concern.
- **Stationarity and linearity**: the standard test assumes stationary, approximately linear dynamics.

For Gaussian variables, Granger causality is equivalent to transfer entropy, an information-theoretic generalisation for nonlinear systems (see [[mutual_information]]).

## Relation to Active Inference
The two meet in the study of brain connectivity. Granger causality estimates directed statistical dependence from data, so it describes functional or directed functional connectivity. Dynamic causal modelling (DCM), developed within the free energy tradition, instead fits an explicit generative model of neuronal dynamics and infers effective connectivity, the influence one region exerts on another under that model. Friston, Moran and Seth (2013) compared the approaches and argued that they answer different questions. Granger methods are model-free and exploratory, whereas DCM requires hypotheses but supports model comparison. Granger causality is not part of Active Inference itself; it is an analysis tool used alongside it.

## Related concepts
- [[causal_inference]] — the wider problem of inferring causes from data
- [[connectivity]] — functional and effective connectivity in the brain
- [[neuroimaging]] — the main setting where the method is applied
- [[mutual_information]] — transfer entropy extends it to directed information flow
- [[stochastic_processes]] — the time-series models on which the test rests

## References
- Granger, C. W. J. (1969). Investigating causal relations by econometric models and cross-spectral methods. Econometrica, 37, 424-438.
- Friston, K., Moran, R., & Seth, A. K. (2013). Analysing connectivity with Granger causality and dynamic causal modelling. Current Opinion in Neurobiology, 23, 172-178.
