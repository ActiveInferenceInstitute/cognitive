---
title: Hebbian Learning
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - learning
  - plasticity
  - synapse
semantic_relations:
  - type: relates
    links:
      - "[[active_inference]]"
      - "[[predictive_coding]]"
      - "[[synaptic_plasticity]]"
---

# Hebbian Learning

## Overview
Hebbian learning is the family of learning rules in which the strength of a connection changes as a function of the correlated activity of the units it connects. It is usually summarized as "cells that fire together wire together", a paraphrase of Donald Hebb's 1949 postulate that repeated participation of one cell in firing another strengthens the connection between them. It is the canonical local, unsupervised rule: a synapse needs only information available at its own pre- and postsynaptic terminals.

## Mechanism
The simplest rate-based form is $\Delta w_{ij} = \eta\, x_j\, y_i$, where $x_j$ is presynaptic activity, $y_i$ postsynaptic activity and $\eta$ a learning rate. Taken alone this rule is unstable, since weights grow without bound and only positive correlations are captured. Standard repairs include:

- **Oja's rule**, $\Delta w_{ij} = \eta\, y_i (x_j - y_i w_{ij})$, which adds a normalizing decay term and makes a single unit converge to the first principal component of its input.
- **BCM theory**, $\Delta w = \eta\, x\, y\,(y - \theta_M)$, in which a sliding postsynaptic threshold $\theta_M$ switches potentiation to depression and stabilizes learning (see [[metaplasticity]]).
- **Spike-timing-dependent plasticity (STDP)**, where the sign of change depends on whether the presynaptic spike precedes or follows the postsynaptic one.

Biologically, the candidate substrate is NMDA-receptor-dependent long-term potentiation, which acts as a coincidence detector of presynaptic glutamate release and postsynaptic depolarization (see [[synaptic_plasticity]]).

## Role in Active Inference
The link is real but specific. In discrete-state active inference, learning of the likelihood mapping is implemented by accumulating Dirichlet concentration parameters, roughly $a \leftarrow a + o \otimes s$, where $o$ is the observation and $s$ the posterior belief over states. That outer-product update is Hebbian in form: it increments an association between co-occurring pre- and postsynaptic quantities. In continuous-state [[predictive_coding]] schemes, gradient descent on [[variational_free_energy]] gives weight changes proportional to a precision-weighted [[prediction_error]] multiplied by presynaptic activity, which is again local and Hebbian-like. The claim is therefore that Hebbian plasticity can be read as a free-energy-minimizing parameter update under a specific generative model, not that the brain is proven to implement it this way. Plain Hebbian rules are purely correlational; the error-driven variants differ because they depend on a mismatch signal, not raw coactivation.

## Related concepts
- [[synaptic_plasticity]] — biological mechanisms (LTP, LTD) that implement Hebbian changes
- [[predictive_coding]] — scheme in which Hebbian-like updates minimize prediction error
- [[prediction_error]] — the signal that gates learning in error-driven variants
- [[unsupervised_learning]] — Hebbian rules are the classic unsupervised learners
- [[homeostatic_plasticity]] — the stabilizing counterpart to Hebbian positive feedback
- [[learning_mechanisms]] — wider catalogue of learning processes

## References
- Hebb, D. O. (1949). *The Organization of Behavior*.
- Oja, E. (1982). A simplified neuron model as a principal component analyzer. *Journal of Mathematical Biology*.
- Bienenstock, E., Cooper, L., & Munro, P. (1982). Theory for the development of neuron selectivity. *Journal of Neuroscience*.
- Friston, K., FitzGerald, T., Rigoli, F., Schwartenbeck, P., & Pezzulo, G. (2017). Active inference: a process theory. *Neural Computation*.
