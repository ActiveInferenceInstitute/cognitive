---
title: Research guides
type: guide
status: stable
updated: 2026-10-10
tags:
  - research
  - methodology
  - active_inference
  - free_energy_principle
semantic_relations:
  - type: relates_to
    links:
      - "[[docs/research/README]]"
      - "[[knowledge_base/cognitive/active_inference]]"
      - "[[knowledge_base/free_energy_principle/README]]"
---

# Research guides

How to carry an Active Inference question through this repository: from a
hypothesis to a seeded, reproducible computation. The package supplies the
discrete, continuous, and homeostatic building blocks listed in
[[docs/api/api_reference|API reference]]; everything else (tasks, data, model
comparison) belongs to the research project.

## Research cycle

1. Formulate the hypothesis in terms of a generative model and free energy.
2. Specify the model. A finite model is the arrays `A`, `B`, `C`, `D`, `E` of a
   `DiscreteGenerativeModel`; see [[docs/guides/application/guide_for_cognitive_modeling|Guide for cognitive modelling]].
3. Derive predictions, for example expected free energy per action
   (`DiscreteGenerativeModel.expected_free_energy` returns total, risk,
   ambiguity, and epistemic gain).
4. Run the simulation with explicit seeds (`InferenceConfig(seed=...)`).
5. Compare behaviour with the predictions, and refine the model.

## Knowledge base entry points

| Area | Page |
|---|---|
| Free Energy Principle | [[knowledge_base/free_energy_principle/README]] |
| Active Inference | [[knowledge_base/cognitive/active_inference]] |
| Predictive processing | [[knowledge_base/cognitive/predictive_processing]] |
| Swarm intelligence | [[knowledge_base/cognitive/swarm_intelligence]] |
| Computational neuroscience | [[knowledge_base/cognitive/computational_neuroscience]] |

## Essential reading

1. Friston (2010), "The free-energy principle: a unified brain theory?", *Nature
   Reviews Neuroscience*.
2. Parr, Pezzulo, and Friston (2022), *Active Inference: The Free Energy
   Principle in Mind, Brain, and Behavior*, MIT Press.

The manuscript bibliography (`docs/manuscript/references.bib`) lists the sources
the repository itself cites.

## Reproducibility

- Record the configuration, the seed, and the package version with each result.
  The manuscript builder writes these to `build_manifest.json`; see
  [`docs/manuscript/06_reproducibility.md`](../manuscript/06_reproducibility.md).
- Build the manuscript with `cognitive-build-manuscript --no-pdf --output build/manuscript`
  to regenerate figures and variables.
- `cognitive-benchmark --repetitions 10` prints timing results as JSON.

## Research documentation

- [[docs/research/README|Research documentation index]]
- [[docs/research/active_inference/applications|Applications]]
- [[docs/research/architectures/multi_agent|Multi-agent architectures]]
- [[docs/research/methodology/README|Methodology]]
- [[docs/guides/learning_paths/catalog_of_learning_paths|Learning path catalog]]
- [[docs/guides/best_practices|Best practices]]
