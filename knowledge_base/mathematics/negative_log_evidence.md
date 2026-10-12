---
title: Negative Log Evidence
description: The negative logarithm of the marginal likelihood of data under a model, equal to surprise in the free energy principle and bounded above by variational free energy.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - model_evidence
  - surprise
  - variational_inference
semantic_relations:
  - type: relates
    links:
      - "[[model_evidence]]"
      - "[[variational_free_energy]]"
      - "[[kl_divergence]]"
      - "[[entropy]]"
      - "[[bayes_theorem]]"
---

# Negative Log Evidence

## Overview
For a generative model $m$ with hidden causes $s$ and observations $o$, the model evidence is the marginal likelihood
$$p(o\mid m)=\int p(o,s\mid m)\,ds ,$$
and the negative log evidence is $-\ln p(o\mid m)$. It is the self-information of the observation under the model, so improbable observations have high values. In Bayesian model comparison the evidence is the quantity that ranks models: a model with higher evidence, equivalently lower negative log evidence, explains the data better after accounting for complexity.

## Relation to variational free energy
The evidence integral is usually intractable, so an approximate posterior $q(s)$ is introduced. Adding and subtracting $\ln p(s\mid o)$ gives the identity
$$F[q]=\mathbb{E}_{q}\big[\ln q(s)-\ln p(o,s\mid m)\big]=-\ln p(o\mid m)+\mathrm{KL}\big[q(s)\,\|\,p(s\mid o,m)\big].$$
Since the divergence is non-negative, variational free energy $F$ is an upper bound on the negative log evidence, equal to it when $q$ equals the true posterior. Minimising $F$ with respect to $q$ therefore improves the approximation to the posterior, and in the same step tightens an estimate of the evidence. The negative of $F$ is the evidence lower bound used in machine learning. The expression can also be written as complexity minus accuracy, $F=\mathrm{KL}[q(s)\|p(s\mid m)]-\mathbb{E}_q[\ln p(o\mid s,m)]$.

Averaging the negative log evidence over outcomes drawn from a process gives the entropy of observations under that process (a cross-entropy if the model differs from the process). The quantity is expressed in nats when natural logarithms are used and in bits for base two.

## Role in Active Inference
In the free energy principle, "surprise" is defined as negative log evidence, $-\ln p(o\mid m)$, with the model $m$ standing for the agent's own expectations. Perception adjusts $q$ to reduce free energy, and action changes $o$ so that observations become less surprising under the model; the long-run average of surprise is the entropy of sensory states, which a persisting system is argued to keep low. This is a technical definition tied to the agent's model and differs from everyday surprise or from information content under the true generating process. The bound relation is a mathematical identity, whereas the interpretation of living systems as minimising surprise is a theoretical claim with a contested empirical status.

## Related concepts
- [[model_evidence]] — the quantity whose negative logarithm is taken
- [[variational_free_energy]] — the upper bound used for inference
- [[kl_divergence]] — the gap between bound and evidence
- [[entropy]] — expected surprise under a distribution
- [[bayes_theorem]] — evidence normalises the posterior
- [[free_energy_principle]] — theory using surprise as the central quantity

## References
- Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.
- Friston, K. (2010). The free-energy principle: a unified brain theory? Nature Reviews Neuroscience, 11, 127-138.
- Cover, T. M., & Thomas, J. A. (2006). Elements of Information Theory (2nd ed.). Wiley.
