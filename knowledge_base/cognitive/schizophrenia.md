---
title: Schizophrenia
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - psychiatry
  - psychosis
  - predictive_coding
semantic_relations:
  - type: relates
    links:
      - "[[computational_psychiatry]]"
      - "[[predictive_processing]]"
      - "[[precision_weighting]]"
      - "[[psychiatric_disorders]]"
---

# Schizophrenia

## Overview
Schizophrenia is a psychiatric disorder characterised by psychosis (hallucinations, delusions, disorganised thought), negative symptoms such as reduced motivation and expression, and cognitive impairment. It is heterogeneous and its causes are multifactorial and incompletely understood. This page summarises computational hypotheses; it is not clinical advice and makes no treatment claims.

## Mechanism
Biological research has implicated dopaminergic and glutamatergic (NMDA receptor) signalling, altered synaptic connectivity, and neurodevelopmental factors. Cognitive findings include deficits in working memory and cognitive control, abnormal sensory gating, reduced mismatch responses in some paradigms, and altered self-monitoring of action. A longstanding idea, aberrant salience, holds that dopamine dysregulation leads irrelevant stimuli to be experienced as significant, which patients then try to explain through delusional beliefs.

## Role in Active Inference
Predictive-processing accounts recast these findings in terms of inference. Hallucinations and delusions are proposed to arise from an imbalance in how prior beliefs and sensory evidence are weighted, so that either unreliable prediction errors drive belief revision or beliefs fail to be constrained by data ([[precision_weighting]], [[prediction_error]]). A related hypothesis ties NMDA-receptor and neuromodulatory abnormalities to errors in estimated precision at multiple levels of a [[hierarchical_inference]] scheme, and failures of sensory attenuation to disturbed sense of agency.

The direction of the proposed abnormality is contested. Some work argues for weakened priors, some for overly strong priors (hallucinations in experimental paradigms have been linked to overweighting of perceptual priors), and some argue the pattern differs by symptom and illness stage. Evidence is mixed, task-dependent, and these accounts are hypotheses under active research, not established causal explanations. They also address mainly positive symptoms; negative symptoms are less well captured.

## Related concepts
- [[computational_psychiatry]] — the modelling field in which these hypotheses are tested
- [[psychiatric_disorders]] — classification and transdiagnostic context
- [[predictive_processing]] — general framework for prior-evidence balance
- [[hierarchical_inference]] — multi-level precision estimation
- [[working_memory]] — a cognitive domain commonly affected
- [[dorsolateral_prefrontal]] — prefrontal region frequently studied in this context
- [[psychiatry]] — free-energy-based applications in detail

## References
- Fletcher, P. C., & Frith, C. D. (2009). Perceiving is believing: a Bayesian approach to explaining the positive symptoms of schizophrenia. *Nature Reviews Neuroscience*.
- Kapur, S. (2003). Psychosis as a state of aberrant salience. *American Journal of Psychiatry*.
- Adams, R. A., Stephan, K. E., Brown, H. R., Frith, C. D., & Friston, K. J. (2013). The computational anatomy of psychosis. *Frontiers in Psychiatry*.
- Sterzer, P., et al. (2018). The predictive coding account of psychosis. *Biological Psychiatry*.
