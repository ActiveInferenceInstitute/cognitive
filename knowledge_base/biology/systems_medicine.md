---
title: Systems Medicine
description: The application of systems biology and network analysis to disease, diagnosis and treatment, including network medicine and P4 medicine, with its challenges and links to computational psychiatry.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - medicine
  - systems_biology
  - networks
semantic_relations:
  - type: relates
    links:
      - "[[systems_biology]]"
      - "[[complex_systems_biology]]"
      - "[[network_theory]]"
      - "[[philosophy_of_medicine]]"
      - "[[computational_psychiatry]]"
wikidata: Q7663871
---

# Systems Medicine

## Overview
Systems medicine applies the approaches of [[systems_biology]] to human health and disease. Instead of attributing a disease to a single faulty gene or pathway, it models how molecular, cellular, organ-level and environmental factors interact, and asks how those interactions become perturbed in illness. Kitano (2002) framed systems biology as the study of networks of genes, proteins and cells as integrated systems, and systems medicine extends this aim to patients and populations.

## Approaches
- **Network medicine**: diseases are mapped onto interaction networks of genes, proteins and metabolites. Barabási, Gulbahce and Loscalzo (2011) proposed that related diseases tend to involve interacting cellular components, so network neighbourhoods can suggest candidate genes and drug targets.
- **Multi-omics integration**: genomic, transcriptomic, proteomic and metabolomic measurements are combined with clinical data to describe individual disease states.
- **Mathematical and computational models**: dynamical models of signalling, metabolism or physiology are used to simulate how interventions could change a patient's trajectory.
- **P4 medicine**: Hood and Flores (2012) described a vision of medicine that is predictive, preventive, personalised and participatory, driven by dense individual data.

## Challenges
Large data sets are noisy and incompletely annotated, and network edges differ across tissues and conditions. Predictive models need validation in prospective clinical studies, and claims about personalised prevention vary widely in their evidence base. Questions also arise about data privacy and how complex model output should inform clinical judgement, which connect to [[philosophy_of_medicine]] and [[evidence_based_medicine]].

## Relation to Active Inference
The connection is mainly through computational psychiatry and physiology. Active Inference models have been used to describe how altered precision or priors could give rise to symptoms in conditions such as schizophrenia and autism, and they offer dynamic, mechanistic hypotheses of the kind systems medicine seeks. This work is an active research programme with limited clinical validation so far. It is a modelling partner to systems medicine, not part of its established methods.

## Related concepts
- [[systems_biology]] — the parent discipline supplying methods
- [[complex_systems_biology]] — complexity-oriented view of biological organisation
- [[network_theory]] — graph methods behind disease networks
- [[evidence_based_medicine]] — the standard against which system-level claims are tested
- [[philosophy_of_medicine]] — conceptual questions about disease and explanation
- [[computational_psychiatry]] — model-based approach to mental disorders

## References
- Kitano, H. (2002). Systems biology: a brief overview. Science, 295, 1662-1664.
- Barabási, A.-L., Gulbahce, N., & Loscalzo, J. (2011). Network medicine: a network-based approach to human disease. Nature Reviews Genetics, 12, 56-68.
- Hood, L., & Flores, M. (2012). A personal view on systems medicine and the emergence of proactive P4 medicine: predictive, preventive, personalized and participatory. New Biotechnology, 29, 613-624.
