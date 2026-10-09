---
title: Structure Learning
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - learning
  - generative_model
  - model_selection
semantic_relations:
  - type: relates
    links:
      - "[[active_inference]]"
      - "[[generative_model]]"
      - "[[model_selection]]"
---

# Structure Learning

## Overview
Structure learning is the acquisition of the form of a model, as opposed to the values of its parameters: which variables exist, how many latent states or factors there are, and which dependencies connect them. Parameter learning fits a given structure; structure learning decides among structures, or builds a new one. In cognitive science it is invoked to explain how people discover categories, causal graphs, and hidden rules from sparse data.

## Mechanism
The standard formalization is Bayesian model comparison. Each candidate structure $m$ is scored by its (approximate) log evidence, and the variational free energy $F_m$ upper-bounds the negative log evidence: $F_m \ge -\ln p(o \mid m)$. Because $F_m$ equals accuracy minus complexity, structure comparison automatically penalizes models that are more elaborate than the data justify. Computational routes include:

- **Bayesian model reduction (BMR)**: given a fitted "full" model, the evidence of nested reduced models with tighter priors can be computed analytically from the full model's posterior, without refitting. Pruning redundant parameters follows directly.
- **Model expansion**: adding a latent state, factor, or hierarchical level when the current model leaves persistent unexplained prediction error.
- **Nonparametric priors** (for example Chinese restaurant process priors) that let the number of latent causes grow with the data.

Structure search is combinatorial, which is why most practical schemes grow or shrink a model incrementally.

## Role in Active Inference
Active inference treats the generative model as the agent's hypothesis about how its observations arise, so structure learning is model selection on that hypothesis space. Smith and colleagues demonstrated a scheme in which an agent starts with a small set of latent states and, when observations are poorly explained, expands the model; BMR then removes redundancy. This provides a free-energy account of concept learning and, speculatively, of sleep or offline "model compression" (see [[pruning]] and [[systems_consolidation]]). The connection is a modeling proposal with working simulations, not an established neural fact. Open problems include how candidate structures are proposed in the first place, since free energy can rank models but does not generate them.

## Related concepts
- [[generative_model]] — the object whose structure is learned
- [[model_selection]] — the comparison machinery structure learning relies on
- [[model_complexity]] — the penalty term that disfavors over-elaborate structures
- [[hierarchical_models]] — structure learning often adds levels or factors
- [[bayesian_inference]] — the underlying logic of comparing structures by evidence
- [[unsupervised_learning]] — structure is typically discovered without labels

## References
- Friston, K., & Penny, W. (2011). Post hoc Bayesian model selection. *NeuroImage*.
- Smith, R., Schwartenbeck, P., Parr, T., & Friston, K. (2020). An active inference approach to modeling structure learning: concept learning as an example case. *Frontiers in Computational Neuroscience*.
- Gershman, S. J., & Niv, Y. (2010). Learning latent structure: carving nature at its joints. *Current Opinion in Neurobiology*.
