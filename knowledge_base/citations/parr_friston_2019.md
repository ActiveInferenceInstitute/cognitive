---
title: "Parr & Friston (2019)"
type: citation
status: stable
created: 2026-02-06
updated: 2026-02-07
tags:
  - citation
  - active-inference
  - generalised-free-energy
  - variational-inference
  - markov-blankets
---

# Parr & Friston (2019)

**Generalised free energy and active inference**

## Reference

Parr, T., & Friston, K. J. (2019). Generalised free energy and active inference. *Biological Cybernetics*, 113(5-6), 495-513.

## Motivation

Prior Active Inference formulations used variational free energy ($F$) for perception and expected free energy ($G$) for planning as separate objectives. This created a theoretical asymmetry: why should perception and planning use different cost functions? This paper resolves that asymmetry.

## Key Contributions

1. **Generalised free energy** $\mathcal{F}$ unifying variational and expected free energy under a single objective
2. **Formal derivation** showing $G$ arises naturally when marginalizing $\mathcal{F}$ over future observations
3. **Temporal depth**: Multi-step policy evaluation as iterated free energy minimization
4. **Risk-sensitive planning**: Natural incorporation of risk aversion via precision over policies
5. **Connection to information geometry**: $\mathcal{F}$ as a natural gradient flow on a statistical manifold

## Core Equations

### Generalised Free Energy

$$\mathcal{F}(\pi) = \underbrace{F(q(s_\tau|\pi), o_{\leq t})}_{\text{past: variational FE}} + \underbrace{\mathbb{E}_{q(o_{>t}|\pi)}[F(q(s_\tau|\pi), o_{>t})]}_{\text{future: expected FE}}$$

### Epistemic-Pragmatic Decomposition

$$G(\pi, \tau) = \underbrace{-\mathbb{E}_{q(o_\tau|\pi)}[D_{KL}[q(s_\tau|o_\tau, \pi)||q(s_\tau|\pi)]]}_{\text{negative epistemic value}} + \underbrace{D_{KL}[q(o_\tau|\pi)||p(o_\tau)]}_{\text{risk (pragmatic value)}}$$

## Theoretical Impact

This paper resolves the perception-planning duality in Active Inference by showing both are minimizations of the same generalised free energy at different temporal scales. It established the theoretical standard used in subsequent computational implementations.

| Aspect | Before (2017) | After (2019) |
| --- | --- | --- |
| Perception objective | Variational $F$ | $\mathcal{F}$ (present) |
| Planning objective | Expected $G$ | $\mathcal{F}$ (future) |
| Theoretical basis | Two objectives | One unified objective |
| Risk sensitivity | Ad hoc | Naturally emerges |

## Related Knowledge Base Articles

- [[knowledge_base/research/concepts/parr_2019]] — Detailed concept article on this paper
- [[knowledge_base/research/concepts/friston_2017]] — Foundational process theory reference

## Related Citations

- [[parr_friston_2017|Parr & Friston (2017)]]
- [[friston_2019_particular|Friston (2019)]]
- [[parr_pezzulo_friston_2022|Parr, Pezzulo & Friston (2022)]]

---

> [!note] Open Source and Licensing
> Repository: [ActiveInferenceInstitute/cognitive](https://github.com/ActiveInferenceInstitute/cognitive)
>
> - Documentation and knowledge base content: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
