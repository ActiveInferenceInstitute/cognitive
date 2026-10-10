---
title: Processing Speed
description: Processing speed is the rate at which simple cognitive operations are executed, measured by reaction time and symbol-coding tasks, and it changes with development and aging.
type: concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - cognition
  - development
  - aging
semantic_relations:
  - type: relates
    links:
      - "[[cognitive_development]]"
      - "[[working_memory]]"
      - "[[evidence_accumulation]]"
---

# Processing Speed

## Overview
Processing speed is the rapidity with which a person can perceive, decide, and respond when performing simple, well-practised cognitive operations. It is typically treated as a limited-capacity resource that constrains higher cognition, and it is a core index in intelligence batteries. The cited pages use it for developmental increases linked to myelination and for slowed processing in language disorders.

## Measurement and change
- **Tasks**: simple and choice reaction time, inspection time (shortest exposure needed to discriminate stimuli), and paper-and-pencil speeded tasks such as symbol-digit coding or letter comparison.
- **Development**: speed rises steeply through childhood and adolescence, and Kail (1991) reported that the developmental trajectory is described by a similar curve across many tasks, suggesting a global rather than task-specific change. Myelination and synaptic pruning are proposed contributors (white-matter maturation is studied with diffusion imaging, see [[neuroimaging]]).
- **Aging**: Salthouse (1996) proposed that slowing is a primary driver of adult age differences in cognition, because limited-time and simultaneity mechanisms degrade performance on complex tasks. The strength of the causal claim is debated, as slowing is statistically entangled with working memory and executive differences ([[working_memory]]).
- **Individual differences**: speed correlates modestly to moderately with general intelligence, and slowing is seen in several neurological and psychiatric conditions, which makes it a sensitive but non-specific marker.
- **Speed-accuracy trade-off**: raw reaction time conflates caution with efficiency. Sequential-sampling models decompose response times into an evidence accumulation rate, a decision threshold, and non-decision time, and the rate is often the component that tracks individual and age differences ([[evidence_accumulation]], [[mathematical_psychology]]).

## Role in Active Inference
The link is to inference dynamics. Belief updating in the framework is a gradient flow with a timescale, so the time to settle on a posterior is bounded by the rate of that flow and by the precision of incoming evidence: more precise evidence yields faster convergence. This parallels the drift-rate interpretation of speed and suggests that individual or age differences could reflect sensory precision, neural gain, or prior strength ([[precision_weighting]], [[neural_efficiency]]). It is an interpretive mapping; the framework does not by itself predict the observed lifespan trajectory, and diffusion-model parameters remain the standard quantitative tool.

## Related concepts
- [[cognitive_development]] — developmental increases in speed
- [[working_memory]] — capacity tightly coupled to speed
- [[evidence_accumulation]] — model-based decomposition of response times
- [[neural_efficiency]] — efficiency of neural signalling
- [[language_processing]] — domain where slowed processing affects outcomes
- [[signal_detection]] — complementary framework for sensitivity and bias

## References
- Kail, R. (1991). Developmental change in speed of processing during childhood and adolescence. *Psychological Bulletin*, 109, 490-501.
- Salthouse, T. A. (1996). The processing-speed theory of adult age differences in cognition. *Psychological Review*, 103, 403-428.
