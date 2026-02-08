---
title: "Friston et al. (2017): Active Inference: A Process Theory"
type: reference
status: stable
created: 2024-02-12
updated: 2026-02-07
tags: [active_inference, process_theory, POMDP, reference, foundational]
semantic_relations:
  - type: foundational_for
    links: [[knowledge_base/cognitive/active_inference]], [[knowledge_base/mathematics/active_inference_pomdp]], [[knowledge_base/mathematics/expected_free_energy]]]
  - type: cites
    links: [[[parr_2019]], [[knowledge_base/mathematics/variational_free_energy]], [[knowledge_base/mathematics/free_energy_principle]]]
---

# Friston et al. (2017): Active Inference — A Process Theory

## Citation

Friston, K. J., FitzGerald, T., Rigoli, F., Schwartenbeck, P., & Pezzulo, G. (2017). Active inference: A process theory. *Neural Computation*, 29(1), 1-49.

## Motivation

Prior to this paper, Active Inference was primarily a theoretical framework. This work provides a concrete computational process theory that specifies exactly how biological (and artificial) agents can implement Active Inference using discrete state-space generative models (POMDPs).

## Key Contributions

1. **Unified process theory** for perception, action, planning, and learning
2. **POMDP generative model** with matrices $A$ (observations), $B$ (transitions), $C$ (preferences), $D$ (initial states)
3. **Expected free energy** $G(\pi)$ as the objective for policy selection
4. **Epistemic-pragmatic decomposition**: $G = -\text{info gain} + \text{expected surprise}$
5. **Neuronal process theory** mapping belief updates to empirical neuronal dynamics

## Core Equations

### Generative Model

```math
p(o_{1:T}, s_{1:T}, \pi) = p(\pi) \prod_{\tau=1}^{T} p(o_\tau|s_\tau) p(s_\tau|s_{\tau-1}, \pi)
```

### Expected Free Energy

```math
G(\pi) = \sum_{\tau=t+1}^{T} G(\pi, \tau) = \sum_\tau \left[ \underbrace{-\mathbb{E}[\ln p(o_\tau|s_\tau)]}_{\text{ambiguity}} + \underbrace{D_{KL}[q(s_\tau|\pi)||p(s_\tau)]}_{\text{risk}} \right]
```

Equivalently decomposed into:

```math
G(\pi, \tau) = \underbrace{-I[o_\tau; s_\tau | \pi]}_{\text{(negative) epistemic value}} + \underbrace{H[q(o_\tau|\pi)] - H[p(o_\tau)]}_{\text{extrinsic value}}
```

### Policy Selection

```math
P(\pi) = \sigma(-\gamma \cdot G(\pi)) = \frac{\exp(-\gamma G(\pi))}{\sum_{\pi'} \exp(-\gamma G(\pi'))}
```

### State Estimation (Variational Message Passing)

```math
\mathbf{s}_\tau = \sigma\left(\ln A^T \mathbf{o}_\tau + \ln B(\pi)^T \mathbf{s}_{\tau-1} + \ln B(\pi) \mathbf{s}_{\tau+1}\right)
```

## Neuronal Process Theory

The paper maps these computations to neuronal dynamics:

| Computational Quantity | Neural Correlate | Brain Region |
| --- | --- | --- |
| Posterior beliefs $q(s)$ | Neuronal firing rates | Cortex |
| Prediction errors $\varepsilon$ | Superficial pyramidal cells | Layer 2/3 |
| Predictions $\hat{o}$ | Deep pyramidal cells | Layer 5/6 |
| Precision $\gamma$ | Neuromodulatory gain | Dopaminergic system |
| Policy evaluation $G(\pi)$ | Striatal activity | Basal ganglia |
| Action selection | Motor commands | Motor cortex |

```mermaid
graph TD
    subgraph "Neural Implementation"
        O[Observations] --> PE[Prediction Errors]
        PE --> BU[Belief Update q(s)]
        BU --> PD[Predictions]
        PD --> PE
        BU --> GPI[G(π) Evaluation]
        GPI --> PS[Policy Selection P(π)]
        PS --> A[Action]
        A --> O
    end
    style PE fill:#fbb,stroke:#333
    style BU fill:#bbf,stroke:#333
    style PS fill:#bfb,stroke:#333
```

## Impact

This paper established the standard computational framework used across subsequent Active Inference research. Key follow-up works include [[parr_2019|Parr & Friston (2019)]] on generalised free energy and Da Costa et al. (2020) on discrete state-space implementations.

## Related Topics

- [[knowledge_base/cognitive/active_inference]] — Core Active Inference framework
- [[knowledge_base/mathematics/active_inference_pomdp]] — POMDP formulation
- [[knowledge_base/mathematics/expected_free_energy]] — Expected free energy derivation
- [[knowledge_base/mathematics/variational_free_energy]] — Variational free energy
- [[knowledge_base/mathematics/free_energy_principle]] — Free energy principle
- [[parr_2019]] — Extension with generalised free energy

## References

- Friston, K. J., et al. (2017). Active inference: A process theory. *Neural Computation*, 29(1), 1-49.
- Da Costa, L., et al. (2020). Active inference on discrete state-spaces.
- Parr, T., & Friston, K. J. (2019). Generalised free energy and active inference.
