---
title: Anomaly Detection
description: The identification of observations that deviate from a model of normal behaviour, covering scoring rules, Gaussian and density-based methods, anomaly types, thresholds, and the connection to surprise in Active Inference.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - statistics
  - novelty_detection
  - surprise
semantic_relations:
  - type: relates
    links:
      - "[[signal_detection]]"
      - "[[information_theory]]"
      - "[[variational_free_energy]]"
      - "[[gaussian_distributions]]"
      - "[[time_series_analysis]]"
wikidata: Q3560506
---

# Anomaly Detection

## Overview
Anomaly detection, also called outlier or novelty detection, is the task of finding observations that do not conform to an expected pattern. A model of normal behaviour is constructed or assumed, each new observation $x$ receives an anomaly score $s(x)$, and observations with $s(x) > \tau$ for some threshold $\tau$ are flagged. Applications include fault detection, fraud screening and physiological monitoring.

## Scores and methods
- **Likelihood-based.** Given a density $p(x)$ fitted to normal data, a natural score is the surprisal $s(x) = -\ln p(x)$. Observations in low-density regions score highly.
- **Gaussian model.** For $x \in \mathbb{R}^d$ with mean $\mu$ and covariance $\Sigma$, the Mahalanobis distance $d^2(x) = (x-\mu)^\top \Sigma^{-1}(x-\mu)$ measures deviation in units of the covariance. If $x$ is truly Gaussian, $d^2$ follows a chi-squared distribution with $d$ degrees of freedom, so the threshold $\tau$ set to its $(1-\alpha)$ quantile gives a false alarm rate of $\alpha$ for normal data.
- **Distance and density methods.** The distance to the $k$-th nearest neighbour, or a local density ratio as in the local outlier factor, gives a score without a parametric model.
- **Reconstruction error.** A model such as principal component analysis or an autoencoder is trained on normal data, and $s(x) = \lVert x - \hat{x} \rVert^2$ is large when the model cannot reproduce $x$.

Chandola, Banerjee and Kumar (2009) distinguish point anomalies, contextual anomalies that are unusual only in a particular context such as time of day, and collective anomalies, where a group of observations is abnormal together. Training setups range from fully supervised, which needs labelled anomalies, through semi-supervised, trained on normal data only, to unsupervised, which assumes anomalies are rare. Because anomalies are rare, evaluation relies on precision-recall or ROC analysis, and the threshold trades false alarms against misses in the manner of [[signal_detection]].

## Relation to Active Inference
In Active Inference an agent's generative model assigns each observation a surprisal $-\ln p(o \mid m)$, and variational free energy is an upper bound on it: $F \geq -\ln p(o \mid m)$. High free energy therefore signals that an observation is poorly explained, which is the same logic as likelihood-based anomaly scoring, and sustained prediction error indicates that the model should be revised. This is a direct mathematical correspondence for the surprisal score. Using an agent's free energy as a deployed detector is an application whose quality depends on the generative model.

## Related concepts
- [[signal_detection]] — the decision-theoretic treatment of thresholds
- [[information_theory]] — the source of surprisal as a score
- [[variational_free_energy]] — the bound on surprisal used by agents
- [[gaussian_distributions]] — the basis of Mahalanobis scoring
- [[time_series_analysis]] — the setting for contextual and collective anomalies
- [[probabilistic_graphical_models]] — models from which normal-data densities are built

## References
- Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly detection: a survey. ACM Computing Surveys, 41(3), Article 15.
- Pimentel, M. A. F., Clifton, D. A., Clifton, L., & Tarassenko, L. (2014). A review of novelty detection. Signal Processing, 99, 215-249.
- Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.
