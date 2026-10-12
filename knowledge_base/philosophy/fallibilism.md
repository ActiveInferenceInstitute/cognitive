---
title: Fallibilism
description: "Fallibilism holds that any belief, however well supported, could be mistaken, and shows how this stance shapes pragmatism, scientific method, and probabilistic belief updating in Active Inference."
type: philosophy
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - philosophy
  - epistemology
  - uncertainty
semantic_relations:
  - type: relates
    links:
      - "[[pragmatism]]"
      - "[[scientific_method]]"
      - "[[bayesian_inference]]"
      - "[[active_inference]]"
wikidata: Q192600
---

# Fallibilism

## Overview

Fallibilism is the epistemological position that no empirical belief can be held with certainty that rules out error. It is compatible with justified belief and with knowledge claims, but denies that justification ever reaches the point where revision becomes impossible. The term is most closely tied to Charles Peirce, who argued against Cartesian certainty and treated inquiry as a self-correcting community practice.

## Core claims

- **Weak fallibilism**: any particular belief might be false, and we cannot always tell which ones. This is nearly uncontroversial.
- **Strong fallibilism**: all beliefs, including logical and mathematical ones, are open to revision in principle. This is contested; many philosophers exempt logic or arithmetic.
- **Fallibilism versus scepticism**: the sceptic concludes that we lack knowledge. The fallibilist keeps knowledge and gives up the demand for infallibility as a condition on it.
- **Fallibilism versus relativism**: admitting error is possible presupposes that there is something to be wrong about, so most fallibilists retain a notion of truth as a regulative ideal.
- **Methodological consequence**: hypotheses should be stated so that evidence could count against them, and inquiry should be organised to detect and correct errors. Karl Popper's falsificationism is a well-known development of this attitude, though Popper himself preferred the label critical rationalism.

## Formal analogue

Probabilistic epistemology gives fallibilism a precise form. A credence strictly between 0 and 1 expresses that the belief is held but revisable; in Bayesian updating, a prior of exactly 0 or 1 can never be moved by any evidence, so assigning such priors is the formal counterpart of claiming infallibility. Using $p(h \mid e) = p(e \mid h)\,p(h)/p(e)$, a hypothesis with $p(h)=1$ stays at 1 whatever $e$ is observed (when $p(e)>0$). Fallibilist modelling therefore keeps all priors and likelihoods away from the extremes, for example by using smoothed or Dirichlet-parameterised distributions.

## Role in Active Inference

Active Inference represents beliefs as probability distributions over hidden states, parameters and policies, and updates them by minimising variational free energy. Every belief carries explicit uncertainty, and a posterior is never treated as final: new observations keep reshaping it. Dirichlet concentration parameters on likelihood and transition matrices make this literal, since counts grow with experience but remain revisable. The match with fallibilism is structural rather than a claim the framework makes about itself; the free-energy principle does not require the philosopher's commitment, and a modeller can still choose near-deterministic priors. The pragmatist reading of the framework (models are judged by their usefulness for prediction and control) fits the Peircean version most naturally.

## Related concepts

- [[pragmatism]] — Peirce's fallibilism is a core pragmatist doctrine
- [[scientific_method]] — error correction and falsification as institutional practice
- [[bayesian_inference]] — graded credences as the formal expression of revisable belief
- [[scientific_realism]] — a realist can be fallibilist about particular theories
- [[instrumentalism]] — evaluates models by usefulness rather than truth
- [[theory_ladenness]] — one reason observations cannot give final certainty
