---
title: Clinical Neuroscience
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
      - "[[knowledge_base/research/brain_theory]]"
      - "[[knowledge_base/cognitive/computational_neuroscience]]"
      - "[[knowledge_base/free_energy_principle/applications/psychiatry]]"
      - "[[knowledge_base/cognitive/precision_weighting]]"
      - "[[knowledge_base/cognitive/clinical_psychology]]"
---

# Clinical Neuroscience

## Overview

Clinical neuroscience studies the mechanisms of neurological and psychiatric disorders and the basis of their treatment. Its difficulty is that diagnostic categories are defined by symptoms, while the underlying mechanisms cut across them. Computational psychiatry, as set out by Friston, Stephan, Montague and Dolan (2014), tries to bridge the two levels by fitting mechanistic models to behaviour and brain data and relating parameters to symptoms.

## Main questions

- **Mechanism.** Which computational process is altered in a given condition, and how it relates to neurochemistry and circuitry.
- **Stratification.** Whether model parameters identify subgroups better than symptom categories.
- **Prediction.** Whether parameters predict course of illness or treatment response.
- **Intervention.** How medication, psychotherapy and brain stimulation change the relevant parameters.

Tools include dynamic causal modelling (Friston, Harrison and Penny, 2003) of effective connectivity in imaging data, and fitting of reinforcement-learning and Bayesian models to behaviour.

## Connection to Active Inference

The connection is an active research programme, in which several hypotheses have been proposed and few have been tested in patients or have changed clinical practice. The common thread is *precision*, the weight placed on prediction errors versus priors. Altered precision weighting can in principle produce a range of symptoms:

- **Psychosis.** Adams and colleagues (2013) described the computational anatomy of psychosis, where aberrant weighting of prior beliefs and sensory evidence could give rise to hallucinations and delusions.
- **Autism.** Pellicano and Burr (2012) proposed that perception is dominated by sensory input because of weak priors, and Lawson, Rees and Friston (2014) proposed that sensory precision is too high. Both are hypotheses, and the empirical results have been inconsistent.
- **Depression and anxiety.** Barrett, Quigley and Hamilton (2016) proposed an active inference account of allostasis and interoception in depression, involving mis-calibrated predictions about bodily state.
- **Movement disorders.** Dopamine is proposed to encode precision, which links Parkinson's disease to reduced confidence in action policies.

The strength of the approach is that it offers a common formalism across disorders and an explicit link between neuromodulators and computation. Its weaknesses are that these accounts are flexible enough to explain opposing findings after the fact, that parameter estimates from behaviour can be unreliable, and that mapping model parameters to a clinical decision has not yet been shown to help patients. They should be treated as hypotheses.

## In this repository

See [[knowledge_base/free_energy_principle/applications/psychiatry]], [[knowledge_base/cognitive/precision_weighting]], [[knowledge_base/cognitive/clinical_psychology]], [[knowledge_base/cognitive/computational_neuroscience]] and [[knowledge_base/research/brain_theory]].

## References

- Friston, K. J., Stephan, K. E., Montague, R., & Dolan, R. J. (2014). Computational psychiatry: the brain as a phantastic organ. *Lancet Psychiatry*, 1, 148-158.
- Adams, R. A., Stephan, K. E., Brown, H. R., Frith, C. D., & Friston, K. J. (2013). The computational anatomy of psychosis. *Frontiers in Psychiatry*, 4, 47.
- Pellicano, E., & Burr, D. (2012). When the world becomes 'too real': a Bayesian explanation of autistic perception. *Trends in Cognitive Sciences*, 16, 504-510.
- Lawson, R. P., Rees, G., & Friston, K. J. (2014). An aberrant precision account of autism. *Frontiers in Human Neuroscience*, 8, 302.
- Barrett, L. F., Quigley, K. S., & Hamilton, P. (2016). An active inference theory of allostasis and interoception in depression. *Philosophical Transactions of the Royal Society B*, 371, 20160011.
- Friston, K. J., Harrison, L., & Penny, W. (2003). Dynamic causal modelling. *NeuroImage*, 19, 1273-1302.

## Related Resources

- [[knowledge_base/research/brain_theory]]
- [[knowledge_base/cognitive/computational_neuroscience]]
