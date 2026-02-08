---
title: Research Guides
type: guide
status: stable
created: 2025-01-01
updated: 2026-02-07
tags:
  - research
  - methodology
  - active_inference
  - free_energy_principle
semantic_relations:
  - type: relates_to
    links:
      - [[docs/research/README]]
      - [[knowledge_base/cognitive/active_inference]]
      - [[knowledge_base/free_energy_principle/README]]
---

# Research Guides

## Overview

This guide provides a structured approach to conducting research within the Active Inference and Free Energy Principle framework. It covers research methodology, literature navigation, experimental design, and documentation standards.

## Research Methodology

### Active Inference Research Cycle

1. **Formulate hypothesis** — express the research question in terms of generative models and free energy
2. **Build generative model** — specify the POMDP or continuous-time model capturing the hypothesis
3. **Derive predictions** — compute expected behavior under the model (e.g., expected free energy landscapes)
4. **Design experiments** — create simulations or empirical studies to test predictions
5. **Analyze results** — compare observed behavior against model predictions
6. **Refine model** — update the generative model based on discrepancies

### Key Research Areas

| Area | Description | Knowledge Base Entry |
|---|---|---|
| Free Energy Principle | Foundational theory of self-organizing systems | [[knowledge_base/free_energy_principle/README]] |
| Active Inference | Action-perception loop and policy selection | [[knowledge_base/cognitive/active_inference]] |
| Predictive Processing | Hierarchical prediction error minimization | [[knowledge_base/cognitive/predictive_processing]] |
| Swarm Intelligence | Collective behavior and stigmergy | [[knowledge_base/cognitive/swarm_intelligence]] |
| Computational Neuroscience | Neural implementations of Bayesian inference | [[knowledge_base/cognitive/computational_neuroscience]] |

## Literature Navigation

### Essential Reading

1. **Friston (2010)** — "The free-energy principle: a unified brain theory?" (*Nature Reviews Neuroscience*)
2. **Parr, Pezzulo, Friston (2022)** — *Active Inference: The Free Energy Principle in Mind, Brain, and Behavior* (Cambridge University Press)
3. **Smith et al. (2022)** — "A step-by-step tutorial on Active Inference" (*Journal of Mathematical Psychology*)
4. **Da Costa et al. (2020)** — "Active inference on discrete state-spaces" (*Journal of Mathematical Psychology*)

### Research Documentation Hub

The repository maintains a structured research documentation hierarchy:

- [[docs/research/README|Research Documentation Index]] — main research hub
- [[docs/research/active_inference/applications|Application Research]] — applied research examples
- [[docs/research/tools|Research Tools]] — analysis and visualization tools
- [[docs/research/architectures/multi_agent|Multi-Agent Architectures]] — multi-agent research

## Experimental Design

### Simulation Studies

When designing simulation experiments:

- **Define metrics explicitly**: free energy, prediction error, reward, convergence time
- **Control variables**: fix all parameters except the ones under study
- **Use multiple seeds**: run each condition with ≥ 10 random seeds for statistical robustness
- **Document configuration**: record all parameters in version-controlled YAML files
- **Validate baselines**: compare against random and optimal policies

### Empirical Studies

When linking Active Inference models to empirical data:

- **Pre-register hypotheses** and model specifications
- **Fit model parameters** using variational Laplace or grid search
- **Report model evidence** (log marginal likelihood) for model comparison
- **Conduct sensitivity analysis** on key parameters

## Documentation Standards

All research contributions should include:

1. **Research summary** — brief description of question, method, and findings
2. **Generative model specification** — complete formal description of the model
3. **Reproducible code** — scripts that regenerate all results from raw data
4. **Results documentation** — figures, tables, and statistical analyses
5. **Theory links** — cross-references to relevant knowledge base entries

## Related Resources

- [[docs/research/README|Research Documentation Hub]] — main research index
- [[docs/guides/learning_paths/catalog_of_learning_paths|Learning Path Catalog]] — educational pathways
- [[knowledge_base/cognitive/README|Cognitive Science KB]] — theoretical foundations
- [[knowledge_base/mathematics/README|Mathematics KB]] — mathematical background
- [[docs/guides/best_practices|Best Practices]] — coding and documentation standards
